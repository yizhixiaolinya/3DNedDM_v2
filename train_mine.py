# train.py
import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import utils
import clip_loss
import datasets
import models

from itertools import product
import numpy as np
import torch.nn.functional as F
import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import random

def make_data_loader(spec, tag=''):
    '''Create data loader'''
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=4, pin_memory=True)
    return loader

def make_data_loaders():
    '''Make data loaders'''
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    '''Prepare training'''
    if config.get('resume') is not None:
        # 加载最近的训练模型状态
        sv_file = torch.load(config['resume'])  # 加载模型、优化器、训练的epoch等状态
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1  # 继续从上次的 epoch 开始
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

        # 更新学习率调度器
        for _ in range(epoch_start - 1):
            lr_scheduler_G.step()
    else:
        # 从头开始训练
        model_G = models.make(config['model_G']).cuda()

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
    """计算每个维度的起始位置，确保覆盖整个图像"""
    indices = []
    for dim in range(3):
        step = int(patch_size[dim] * (1 - overlap_ratio))
        if step <= 0:
            step = 1  # 防止步长为0
        dim_indices = list(range(0, target_size[dim] - patch_size[dim] + 1, step))
        if len(dim_indices) == 0 or dim_indices[-1] + patch_size[dim] < target_size[dim]:
            dim_indices.append(target_size[dim] - patch_size[dim])
        indices.append(dim_indices)
    return list(product(*indices))  # 所有维度的组合

def img_pad(img, target_shape):
    current_shape = img.shape
    pads = [(0, max(0, target_shape[i] - current_shape[i])) for i in range(len(target_shape))]
    padded_img = np.pad(img, pads, mode='constant', constant_values=0)
    current_shape_2 = padded_img.shape
    crops = []
    for i in range(len(target_shape)):
        if current_shape_2[i] > target_shape[i]:
            crops.append(
                slice((current_shape_2[i] - target_shape[i]) // 2, (current_shape_2[i] + target_shape[i]) // 2))
        else:
            crops.append(slice(None))
    cropped_img = padded_img[tuple(crops)]
    return cropped_img

def slice_image(img, patch_size, overlap=0.25):
    """将整张3D图像切分为patch，确保所有patch尺寸一致"""
    patches = []
    positions = []
    patch_indices = calculate_patch_index(img.shape[1:], patch_size, overlap_ratio=overlap)
    for (i, j, k) in patch_indices:
        patch = img[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
        for l in range(patch.shape[0]):
            patch[l] = img_pad(patch[l], patch_size)
        patches.append(torch.tensor(patch))
        positions.append((i, j, k))# not considering the batch? TODO

    patches_tensor = torch.stack(patches).float()
    return patches_tensor, positions

def reconstruct_image(patches, positions, imgs_shape, patch_size, overlap=0.25):
    """将预测的patch重新拼接成完整的3D图像"""
    # 初始化重构图像
    recon = torch.zeros(imgs_shape)  # recon 和 count 保持在 CPU 上
    count = torch.zeros(imgs_shape)  # 记录每个位置被覆盖的次数

    for patch, (i, j, k) in zip(patches, positions):
        # 确保 patch 不进入 cuda，保持在 CPU 上
        # 添加 patch 到重构图像
        recon[:,i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += patch
        count[:,i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1

    recon /= count  # 处理重叠区域的平均值
    return recon

def get_non_zero_random_crop(src_img, tgt_img, src2tgt_img, tgt2src_img, crop_size):
    """
    从源图像和目标图像中获取随机裁剪的块。

    Args:
    - src_img: 源图像，四维数组 [batch, H, W, D]
    - tgt_img: 目标图像，四维数组 [batch, H, W, D]
    - src2tgt_img: 预测图像
    - tgt2src_img: 预测图像
    - crop_size: 裁剪块的大小，默认为32

    Returns:
    - cropped_src: 裁剪后的源图像块
    - cropped_tgt: 裁剪后的目标图像块
    - cropped_src2tgt: 裁剪后的源到目标图像块
    - cropped_tgt2src: 裁剪后的目标到源图像块
    """
    batch_size, h, w, d = src_img.shape

    # 随机选择裁剪起始位置
    h0 = random.randint(0, max(0, h - crop_size[0]))
    w0 = random.randint(0, max(0, w - crop_size[1]))
    d0 = random.randint(0, max(0, d - crop_size[2]))

    # 裁剪大小为crop_size的块
    cropped_src = src_img[:, h0:h0 + crop_size[0], w0:w0 + crop_size[1], d0:d0 + crop_size[2]]
    cropped_tgt = tgt_img[:, h0:h0 + crop_size[0], w0:w0 + crop_size[1], d0:d0 + crop_size[2]]
    cropped_src2tgt = src2tgt_img[:, h0:h0 + crop_size[0], w0:w0 + crop_size[1], d0:d0 + crop_size[2]]
    cropped_tgt2src = tgt2src_img[:, h0:h0 + crop_size[0], w0:w0 + crop_size[1], d0:d0 + crop_size[2]]

    return cropped_src, cropped_tgt, cropped_src2tgt, cropped_tgt2src

def crop_img(images,crop_size):
    H, W, D = images.shape[1], images.shape[2], images.shape[3]

    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2
    start_d = (D - crop_size) // 2

    # Perform the cropping by slicing
    cropped_images = images[:,
                     start_h:start_h + crop_size,
                     start_w:start_w + crop_size,
                     start_d:start_d + crop_size]
    return cropped_images

def train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, writer, epoch):
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)

    model_G.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.L1Loss()
    clip_loss_fn = clip_loss.CLIPLoss(device=device, lambda_direction=1.0, lambda_global=0.8)

    loss_0 = utils.Averager()
    loss_1 = utils.Averager()
    clip_loss_s2t = utils.Averager()
    clip_loss_t2s = utils.Averager()

    # 累积每个 batch 的 total loss
    epoch_loss_total = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        src_imgs = batch['src_img']
        tgt_imgs = batch['tgt_img']
        src_imgs = crop_img(src_imgs,96)
        tgt_imgs = crop_img(tgt_imgs, 96)
        src_seq = batch['seq_src'].cuda().float()
        tgt_seq = batch['seq_tgt'].cuda().float()

        optimizer_G.zero_grad()

        # 切分图像为 patch
        src_patches, positions = slice_image(src_imgs.cpu().numpy(), patch_size, overlap_ratio)
        tgt_patches, _ = slice_image(tgt_imgs.cpu().numpy(), patch_size, overlap_ratio)

        num_patches = src_patches.size(0)

        # 收集输出结果
        pre_patches_src2tgt_list = []
        pre_patches_tgt2src_list = []

        # 处理每个patch的预测
        for start_idx in range(0, num_patches):
            
            # patches：[k,2,32,32,32]
            # 获取当前批次的 patches
            src_patches_batch = src_patches[start_idx]
            tgt_patches_batch = tgt_patches[start_idx]

            src_seq = src_seq.reshape(src_seq.shape[0], 1, src_seq.shape[-1])
            tgt_seq = tgt_seq.reshape(tgt_seq.shape[0], 1, tgt_seq.shape[-1])
            pre_patches_batch_src2tgt, pre_patches_batch_tgt2src = model_G(src_patches_batch.cuda(), tgt_patches_batch.cuda(), src_seq, tgt_seq)

            if isinstance(pre_patches_batch_src2tgt, tuple):
                pre_patches_batch_src2tgt = pre_patches_batch_src2tgt[0]

            if isinstance(pre_patches_batch_tgt2src, tuple):
                pre_patches_batch_tgt2src = pre_patches_batch_tgt2src[0]

            pre_patches_src2tgt_list.append(pre_patches_batch_src2tgt.cpu())
            pre_patches_tgt2src_list.append(pre_patches_batch_tgt2src.cpu())
        pre_patches_src2tgt = torch.cat(pre_patches_src2tgt_list, dim=0).cpu()
        pre_patches_tgt2src = torch.cat(pre_patches_tgt2src_list, dim=0).cpu()

        # 将所有的 patches 拼接成完整图像
        pre_img_src2tgt = reconstruct_image(pre_patches_src2tgt, positions, src_imgs.shape, patch_size, overlap_ratio)
        pre_img_tgt2src = reconstruct_image(pre_patches_tgt2src, positions, tgt_imgs.shape, patch_size, overlap_ratio)
        # print('pre_img.shape:', pre_img_src2tgt.shape) # torch.Size([2, 96, 96, 96])
        # print('text.shape:', src_seq.shape) # torch.Size([2, 1, 768])

        cropped_src, cropped_tgt, cropped_src2tgt, cropped_tgt2src = get_non_zero_random_crop(src_imgs.cpu(),
                                                                                              tgt_imgs.cpu(),
                                                                                              pre_img_src2tgt.cpu(),
                                                                                              pre_img_tgt2src.cpu(),
                                                                                              patch_size)

        cropped_src = cropped_src2tgt.unsqueeze(1).cuda().float()
        cropped_tgt = cropped_tgt.unsqueeze(1).cuda().float()
        cropped_src2tgt = cropped_src2tgt.unsqueeze(1).cuda().float()
        cropped_tgt2src = cropped_tgt2src.unsqueeze(1).cuda().float()


        # For the first 30 epochs, calculate only L1 loss pre_patches_batch_src2tgt, pre_patches_batch_tgt2src
        if epoch <= 30:
            loss_src2tgt = loss_fn(cropped_src2tgt.cuda(), cropped_tgt.cuda())
            loss_tgt2src = loss_fn(cropped_tgt2src.cuda(), cropped_src.cuda())
            loss_total = loss_src2tgt * 0.5 + loss_tgt2src * 0.5
            loss_0.add(loss_src2tgt.item())  # 对应于模型生成的源目标预测与真实目标的损失（loss_src），即源图像经过模型生成后与目标图像的损失
            loss_1.add(loss_tgt2src.item())

            # 记录每个 batch 的 L1 loss 到 TensorBoard
            writer.add_scalar('Loss/src2tgt', loss_src2tgt.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/tgt2src', loss_tgt2src.item(), epoch * len(train_loader) + batch_idx)
            # print(type(loss_total))
            # print(loss_total)
            # print(loss_total.shape)

            loss_total.backward()

        # From epoch 31 onwards, calculate both L1 and CLIP losses
        else:
            loss_src2tgt = loss_fn(cropped_src2tgt, cropped_tgt)
            loss_tgt2src = loss_fn(cropped_tgt2src, cropped_src)
            loss_clip_src2tgt = clip_loss_fn(src_imgs, src_seq, pre_img_src2tgt, tgt_seq)
            loss_clip_tgt2src = clip_loss_fn(tgt_imgs, tgt_seq, pre_img_tgt2src, src_seq)
            loss_total = (loss_src2tgt + loss_tgt2src) * 0.5 + (loss_clip_src2tgt + loss_clip_tgt2src) * 0.2

            loss_total.backward()

            loss_0.add(loss_src2tgt.item())
            loss_1.add(loss_tgt2src.item())
            clip_loss_s2t.add(loss_clip_src2tgt.item())
            clip_loss_t2s.add(loss_clip_tgt2src.item())

            # 记录每个 batch 的 loss 到 TensorBoard
            writer.add_scalar('Loss/src2tgt', loss_src2tgt.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/tgt2src', loss_tgt2src.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/cliploss_src2tgt', clip_loss_s2t.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/cliploss_tgt2src', clip_loss_t2s.item(), epoch * len(train_loader) + batch_idx)

        # print('epoch:', epoch)
        # print('epoch * len(train_loader) + batch_idx:', epoch * len(train_loader) + batch_idx)
        optimizer_G.step()
        # 累加每个 batch 的 loss_total
        epoch_loss_total += torch.sum(loss_total)

    # 返回整个epoch的累积损失
    return epoch_loss_total.item()

def main(config_, save_path):
    global config, log
    config = config_  # config_为config.yaml文件中的内容

    # 从配置中获取是否询问用户
    ask_user = config.get('ask_user', False)
    # print('ask_user:', ask_user)

    # 当 resume 不为空时，不删除已有路径，保留日志和其他文件
    if config.get('resume') is not None:
        remove_save_path = False
    else:
        # 如果不从检查点继续训练，可以根据需要决定是否删除
        remove_save_path = config.get('remove_save_path', True)  # 从配置中获取，默认删除
    
    # print('remove_save_path:', remove_save_path,flush=True)
    # 设置保存路径，传递 remove 和 ask_user 参数
    log, writer = utils.set_save_path(save_path, remove=remove_save_path, ask_user=ask_user)

    # 初始化 TensorBoard
    # writer 已经在 set_save_path 中初始化

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    patch_size = config['training_params']['patch_size']
    overlap_ratio = config['training_params']['overlap_ratio']

    train_loader, val_loader = make_data_loaders()
    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training()

    # 在 `epoch_start` 变量定义后打印
    # print('epoch_start:', epoch_start)
    resumed_epoch = epoch_start  # 继续训练的起始 epoch

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    # 初始化两个最大 PSNR 变量
    max_val_v_src2tgt = -1e18
    max_val_v_tgt2src = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        optimizer_G.param_groups[0]['lr'] = 0.00001
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        log_info.append('lr_G={:.6f}'.format(optimizer_G.param_groups[0]['lr']))

        # 传递 writer
        train_loss = train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, writer, epoch)

        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        log_info.append('total_loss={:.4f}'.format(train_loss))

        model_G_ = model_G.module if isinstance(model_G, nn.parallel.DistributedDataParallel) else model_G

        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G_.state_dict()
        optimizer_G_spec = config['optimizer_G']
        optimizer_G_spec['sd_G'] = optimizer_G.state_dict()
        sv_file = {'model_G': model_G_spec, 'optimizer_G': optimizer_G_spec, 'epoch': epoch}

        # 保存最新的 epoch-last.pth
        
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        # 只有当 epoch 超过上次中断时，才保存 epoch-{epoch}.pth
        if (epoch_save is not None) and (epoch % epoch_save == 0) and (epoch > resumed_epoch):
            checkpoint_path = os.path.join(save_path, f'epoch-{epoch}.pth')
            # 检查是否已经存在该 epoch 的文件
            if not os.path.exists(checkpoint_path):
                torch.save(sv_file, checkpoint_path)

        # 评估并保存最佳模型
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res_src2tgt, val_res_tgt2src = utils.eval_psnr(val_loader, model_G_)
            log_info.append('val_src2tgt: psnr={:.4f}'.format(val_res_src2tgt))
            log_info.append('val_tgt2src: psnr={:.4f}'.format(val_res_tgt2src))

            # 检查并保存最佳 src2tgt 模型
            if val_res_src2tgt > max_val_v_src2tgt:
                max_val_v_src2tgt = val_res_src2tgt
                torch.save(sv_file, os.path.join(save_path, 'epoch-best_src2tgt.pth'))

            # 检查并保存最佳 tgt2src 模型
            if val_res_tgt2src > max_val_v_tgt2src:
                max_val_v_tgt2src = val_res_tgt2src
                torch.save(sv_file, os.path.join(save_path, 'epoch-best_tgt2src.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

    # 关闭 writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home_data/home/linxin2024/code/3DMedDM_v2/configs/train_lccd_sr.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    ask_user = config.get('ask_user', False)  # 默认值为 False
    batch_size = config['train_dataset']['batch_size']

    save_name = args.name or '_' + args.config.split('/')[-1][:-len('.yaml')] + '_batch_size_' + str(batch_size)
    if args.tag:
        save_name += '_' + args.tag + '_batch_size_' + str(batch_size)

    save_path = os.path.join('/public_bme/data/linxin_debug/Loss', save_name)
    save_path = os.path.join(save_path,datetime.now().strftime("%Y%m%d_%H%M%S"))
    # print('save_path:', save_path)

    os.makedirs(save_path, exist_ok=True)

    main(config, save_path)
