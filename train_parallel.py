# train.py
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
import utils
import clip_loss
import datasets
import models
from itertools import product
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler


def setup_distributed():
    """初始化分布式环境。"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """销毁分布式环境。"""
    dist.destroy_process_group()


def make_data_loader(spec, tag='', is_distributed=False):
    """创建数据加载器。"""
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    if is_distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # 分布式采样器会自动打乱数据
    else:
        sampler = None
        shuffle = (tag == 'train')
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=shuffle, sampler=sampler, num_workers=4, pin_memory=True)
    return loader


def make_data_loaders(is_distributed=False):
    """创建训练和验证数据加载器。"""
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', is_distributed=is_distributed)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', is_distributed=is_distributed)
    return train_loader, val_loader


def prepare_training(local_rank):
    """准备训练，包括模型、优化器、学习率调度器等。"""
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'], map_location='cuda:{}'.format(local_rank))
        model_G = models.make(sv_file['model_G'], load_sd=True).to(local_rank)
        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1  # 获取起始 epoch
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

            for _ in range(epoch_start - 1):
                lr_scheduler_G.step()  # 更新学习率
    else:
        model_G = models.make(config['model_G']).to(local_rank)
        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), config['optimizer_G'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

    log('model_G: #params={}'.format(utils.compute_num_params(model_G, text=True)))
    return model_G, optimizer_G, epoch_start, lr_scheduler_G


def calculate_patch_index(target_size, patch_size, overlap_ratio=0.25):
    """计算每个维度的起始位置，确保覆盖整个图像。"""
    indices = []
    for dim in range(3):
        step = int(patch_size[dim] * (1 - overlap_ratio))
        dim_indices = list(range(0, target_size[dim] - patch_size[dim] + 1, step))
        if dim_indices[-1] + patch_size[dim] < target_size[dim]:
            dim_indices.append(target_size[dim] - patch_size[dim])
        indices.append(dim_indices)
    return list(product(*indices))  # 所有维度的组合


def slice_image(img, patch_size, overlap=0.25):
    """将整张3D图像切分为patch，确保所有patch尺寸一致。"""
    patches = []
    positions = []
    patch_indices = calculate_patch_index(img.shape, patch_size, overlap_ratio=overlap)

    for (i, j, k) in patch_indices:
        patch = img[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
        if patch.shape != tuple(patch_size):
            # 对不足的部分进行填充
            padding = [(0, patch_size[d] - patch.shape[d]) for d in range(3)]
            patch = np.pad(patch, padding, mode='constant', constant_values=0)
            print(f"Padded patch at ({i}, {j}, {k}) to shape {patch.shape}")
        patches.append(torch.tensor(patch))
        positions.append((i, j, k))

    patches_tensor = torch.stack(patches).unsqueeze(1).float()  # [num_patches, 1, D, H, W]
    print(f"Patches tensor shape: {patches_tensor.shape}")  # 调试信息
    return patches_tensor, positions


def reconstruct_image(patches, positions, img_shape, patch_size, overlap=0.25):
    """将预测的patch重新拼接成完整的3D图像。"""
    recon = torch.zeros(img_shape).cuda()
    count = torch.zeros(img_shape).cuda()
    for patch, (i, j, k) in zip(patches, positions):
        recon[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += patch.squeeze(1)
        count[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1
    recon /= count
    return recon


def compute_clip_loss(original_img, pred_img, seq_src, seq_tgt, clip_loss_fn):
    """计算CLIPLoss。"""
    # original_img: [D, H, W] -> [1, 1, D, H, W]
    # pred_img: [D, H, W] -> [1, 1, D, H, W]
    print('original_img.shape:', original_img.shape)  # ([D, H, W])
    print('pred_img.shape:', pred_img.shape)
    print('seq_src.shape:', seq_src.shape)
    print('seq_tgt.shape:', seq_tgt.shape)
    loss = clip_loss_fn(original_img.unsqueeze(0).unsqueeze(0), seq_src, pred_img.unsqueeze(0).unsqueeze(0), seq_tgt)
    return loss


def train(train_loader, model_G, optimizer_G, scaler, local_rank):
    model_G.train()

    # 初始化 CLIPLoss
    lambda_direction = config.get('lambda_direction', 1.0)
    lambda_global = config.get('lambda_global', 0.8)
    clip_loss_fn = clip_loss.CLIPLoss(device=local_rank, lambda_direction=lambda_direction, lambda_global=lambda_global)

    clip_loss_total = utils.Averager()

    patch_size = config.get('patch_size', [32, 32, 32])
    overlap_ratio = config.get('overlap_ratio', 0.25)

    accumulation_steps = config.get('accumulation_steps', 1)

    optimizer_G.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        # 将数据移动到GPU并转换类型
        src_imgs = batch['src_img'].float()  # [batch_size, D, H, W]
        tgt_imgs = batch['tgt_img'].float()  # [batch_size, D, H, W]
        seq_src = batch['seq_src'].float()   # [batch_size, seq_len]
        seq_tgt = batch['seq_tgt'].float()   # [batch_size, seq_len]
        print('src_imgs.shape:', src_imgs.shape)
        print('tgt_imgs.shape:', tgt_imgs.shape)
        print('seq_src.shape:', seq_src.shape)
        print('seq_tgt.shape:', seq_tgt.shape)

        for i in range(src_imgs.size(0)):
            src_img = src_imgs[i].to(local_rank)  # [D, H, W]
            tgt_img = tgt_imgs[i].to(local_rank)  # [D, H, W]
            src_seq = seq_src[i].unsqueeze(0).to(local_rank)  # [1, seq_len]
            tgt_seq = seq_tgt[i].unsqueeze(0).to(local_rank)  # [1, seq_len]

            # 切分图像为patch
            src_patches, positions = slice_image(src_img.cpu().numpy(), patch_size, overlap_ratio)
            tgt_patches, _ = slice_image(tgt_img.cpu().numpy(), patch_size, overlap_ratio)

            # 将patches移动到GPU
            src_patches = src_patches.to(local_rank)
            tgt_patches = tgt_patches.to(local_rank)
            print('src_patches.shape:', src_patches.shape)
            print('tgt_patches.shape:', tgt_patches.shape)

            # 调整文本嵌入的维度
            src_seq_repeated = src_seq.repeat(src_patches.size(0), 1, 1)  # [num_patches, 1, seq_len]
            tgt_seq_repeated = tgt_seq.repeat(tgt_patches.size(0), 1, 1)  # [num_patches, 1, seq_len]
            print('src_seq_repeated.shape:', src_seq_repeated.shape)

            # 使用自动混合精度
            with autocast():
                # 模型预测
                pre_patches = model_G(src_patches, tgt_patches, src_seq_repeated, tgt_seq_repeated)

                # 如果模型返回的是一个元组，提取主要输出
                if isinstance(pre_patches, tuple):
                    pre_patches = pre_patches[0]

                # 重构完整图像
                pre_img = reconstruct_image(pre_patches, positions, src_img.shape, patch_size, overlap_ratio)
                print('pre_img.shape:', pre_img.shape)

                # 计算CLIPLoss
                loss_clip = compute_clip_loss(src_img, pre_img, src_seq, tgt_seq, clip_loss_fn)
                print('loss_clip:', loss_clip)

            # 使用 scaler 进行梯度缩放
            scaler.scale(loss_clip / accumulation_steps).backward()

            # 累加损失
            clip_loss_total.add(loss_clip.item())

            # 梯度累积
            if ((batch_idx * src_imgs.size(0) + i + 1) % accumulation_steps == 0) or \
               ((batch_idx == len(train_loader) - 1) and (i == src_imgs.size(0) - 1)):
                scaler.step(optimizer_G)
                scaler.update()
                optimizer_G.zero_grad()

            # 释放显存
            del src_img, tgt_img, src_seq, tgt_seq, src_patches, tgt_patches, pre_patches, pre_img
            torch.cuda.empty_cache()

    return clip_loss_total.item()


def main(config_, save_path):
    global config, log
    config = config_  # 从配置文件加载的配置
    log, _ = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 检查是否为分布式训练
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_distributed:
        local_rank = setup_distributed()
    else:
        local_rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = make_data_loaders(is_distributed=is_distributed)
    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training(local_rank)

    if is_distributed:
        model_G = DDP(model_G, device_ids=[local_rank], output_device=local_rank)
    else:
        model_G = model_G.to(local_rank)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    # 初始化混合精度训练
    scaler = GradScaler()

    for epoch in range(epoch_start, epoch_max + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)  # 确保每个 epoch 数据不同

        t_epoch_start = timer.t()
        if lr_scheduler_G is not None:
            lr = lr_scheduler_G.get_last_lr()[0]
        else:
            lr = optimizer_G.param_groups[0]['lr']
        log_info = [f'epoch {epoch}/{epoch_max}']
        log_info.append(f'lr_G={lr:.6f}')

        # 调用修改后的 train 函数
        train_loss = train(train_loader, model_G, optimizer_G, scaler, local_rank)
        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        log_info.append(f'clip_loss={train_loss:.4f}')

        if is_distributed:
            model_G_ = model_G.module
        else:
            model_G_ = model_G

        # 仅在主进程保存模型和打印信息
        if not is_distributed or dist.get_rank() == 0:
            # 检查模型参数
            print("Model parameters:")
            for name, param in model_G_.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.data.shape}")

            model_G_spec = config['model_G']
            model_G_spec['sd_G'] = model_G_.state_dict()
            optimizer_G_spec = config['optimizer_G']
            optimizer_G_spec['sd_G'] = optimizer_G.state_dict()
            sv_file = {
                'model_G': model_G_spec,
                'optimizer_G': optimizer_G_spec,
                'epoch': epoch
            }

            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                           os.path.join(save_path, f'epoch-{epoch}.pth'))

            if (epoch_val is not None) and (epoch % epoch_val == 0):
                val_res = utils.eval_psnr(val_loader, model_G_)
                log_info.append(f'val: psnr={val_res:.4f}')

                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append(f'{t_epoch} {t_elapsed}/{t_all}')

            log(', '.join(log_info))

    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home_data/home/linxin2024/code/3DMedDM_v2/configs/train_lccd_sr.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    # 设置 GPU（在分布式训练中，不需要手动设置 CUDA_VISIBLE_DEVICES）
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # 获取 batch_size 值
    batch_size = config['train_dataset']['batch_size']

    # 根据 batch_size 动态生成保存路径
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')] + '_batch_size_' + str(batch_size)
    if args.tag is not None:
        save_name += '_' + args.tag + '_batch_size_' + str(batch_size)
    print('save_name:', save_name)

    # 设置保存路径：存储到 `save/train/CLIPLoss` 目录
    save_path = os.path.join('./save/train/CLIPLoss', save_name)

    # 确保路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始训练
    main(config, save_path)
