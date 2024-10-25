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
        sv_file = torch.load(config['resume'])  # 加载训练的模型状态(包括模型、优化器、训练的epoch等)
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1  # get epoch
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

        for _ in range(epoch_start - 1):
            lr_scheduler_G.step()  # update lr
    else:
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

def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    original_spacing = img_ref.GetSpacing()
    original_size = img_ref.GetSize()
    resample = sitk.ResampleImageFilter()
    img = sitk.GetImageFromArray(vol)
    resample.SetOutputOrigin(img_ref.GetOrigin())
    resample.SetOutputDirection(img_ref.GetDirection())
    if new_spacing is None:
        resample.SetOutputSpacing(img_ref.GetSpacing())
    else:
        resample.SetOutputSpacing(tuple(new_spacing))
    size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    resample.SetSize(size)
    newimage = resample.Execute(img)
    sitk.WriteImage(newimage, out_path)

def set_new_spacing(ori_spacing, coord_size, crop_size):
    scale0 = coord_size[0] / crop_size[0]
    scale1 = coord_size[1] / crop_size[1]
    scale2 = coord_size[2] / crop_size[2]
    new_spacing = (ori_spacing[0] / scale0, ori_spacing[1] / scale1, ori_spacing[2] / scale2)
    return new_spacing

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

def calculate_patch_index(target_size, patch_size, overlap_ratio=0.25):
    shape = target_size

    gap = int(patch_size[0] * (1 - overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0] - patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1] - patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2] - patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos

def _get_pred(crop_size, overlap_ratio, model, img_vol_0, img_vol_1, coord_size, coord_hr, seq_src, seq_tgt):
    """获取预测结果"""
    print('img.shape:', img_vol_0.shape)  # ([2, 144, 192, 192])
    print('seq.shape:', seq_src.shape) # seq.shape: torch.Size([2, 1, 768])

    W, H, D = img_vol_0.shape[-3:]  # 获取长宽高
    print('whd:',W ,H ,D)  # 192 192 192
    # crop_size 表示每次裁剪的3D图像块的大小
    W_po, H_po, D_po = crop_size[0], crop_size[1], crop_size[2]
    # coord_size 是在高分辨率图像下的目标块尺寸，通常用于将低分辨率图像与高分辨率坐标进行对齐
    W_pt, H_pt, D_pt = coord_size[0], coord_size[1], coord_size[2]
    # 计算缩放比例
    scale0 = W_pt / W_po
    scale1 = H_pt / H_po
    scale2 = D_pt / D_po
    # 计算高分辨率的图像尺寸
    W_t = int(W * scale0)
    H_t = int(H * scale1)
    D_t = int(D * scale2)
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    # 生成预测矩阵
    pred_0_1 = np.zeros((W_t, H_t, D_t))
    pred_1_0 = np.zeros((W_t, H_t, D_t))
    freq_rec = np.zeros((W_t, H_t, D_t))  # 用于记录每个位置预测的次数，方便在最终进行平均
    start_time = time.time()

    for start_pos in pos:
        # 通过分割图像为小块（patch），然后对每个小块进行预测，最后将所有小块的预测结果组合回完整的高分辨率图像中
        # 提取低分辨率图像块
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1],
                         start_pos[2]:start_pos[2] + crop_size[2]]
        img_1_lr_patch = img_vol_1[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1],
                         start_pos[2]:start_pos[2] + crop_size[2]]
        # print('img_0_lr_patch.shape:', img_0_lr_patch.shape) # img_0_lr_patch.shape: torch.Size([2, 32, 32, 192])
        img_0_lr_patch = torch.tensor(img_0_lr_patch).cuda().float()
        img_1_lr_patch = torch.tensor(img_1_lr_patch).cuda().float()
        print('img_0_lr_patch.shape:', img_0_lr_patch.shape)  # [2, 20, 60, 60] ->(crop_size修改后) [2, 8, 32, 32]
        print('img_1_lr_patch.shape:', img_1_lr_patch.shape)

        # 将两个图像块 img_0_lr_patch 和 img_1_lr_patch 以及相应的文本嵌入 seq_src 和 seq_tgt 传入模型进行预测，生成预测结果 pred_0_1_patch
        pred_0_1_patch = model(img_0_lr_patch, img_1_lr_patch, seq_src.cuda().float(), seq_tgt.cuda().float())

        # 获取主输出
        pred_0_1_patch = pred_0_1_patch[0]

        # 使用 reshape(W_pt, H_pt, D_pt) 将预测结果重新塑形为高分辨率图像块的尺寸
        # pred_0_1_patch = pred_0_1_patch.squeeze(0).cpu().numpy().reshape(W_pt, H_pt, D_pt)
        # print('pred_0_1_patch:', pred_0_1_patch)

        print(f"Original pred_0_1_patch shape before squeezing: {pred_0_1_patch.shape}") # torch.Size([2, 1, 32, 32, 192])

        # 如果 batch_size 为 2，需要逐个处理每个 batch
        # 去掉 channel 维度
        pred_0_1_patch = pred_0_1_patch.squeeze(1).detach().cpu().numpy()  # 去掉 channel 维度
        print('pred_0_1_patch',pred_0_1_patch.shape) # (2, 32, 32, 32)

        # 处理 batch_size 维度
        for batch_idx in range(pred_0_1_patch.shape[0]):
            single_patch = pred_0_1_patch[batch_idx]  # 提取单个 batch
            # print(f"Single patch shape: {single_patch.shape}")

            # 检查是否需要 reshape，并进行处理
            if single_patch.shape == (W_pt, H_pt, D_pt):
                print("No reshape needed.")
            else:
                try:
                    single_patch = single_patch.reshape(W_pt, H_pt, D_pt)
                    print(f"Reshaped patch shape: {single_patch.shape}")
                except ValueError as e:
                    print(f"Error in reshaping: {e}")
                    print(f"Cannot reshape array of size {single_patch.size} into shape ({W_pt}, {H_pt}, {D_pt})")

        # 将结果重新拼接回完整的图像中
        target_pos0 = int(start_pos[0] * scale0)
        target_pos1 = int(start_pos[1] * scale1)
        target_pos2 = int(start_pos[2] * scale2)

        # 确保维度匹配
        pred_0_1[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt,
        target_pos2:target_pos2 + D_pt] += single_patch

        # 记录每个位置的预测次数
        freq_rec[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt, target_pos2:target_pos2 + D_pt] += 1

    end_time = time.time()
    print(end_time - start_time)

    # 计算预测的最终输出
    pred_0_1_img = pred_0_1 / freq_rec
    # print('pred_0_1_img:',pred_0_1_img)

    return pred_0_1_img

def compute_clip_loss(original_img, pred_img, seq_src, seq_tgt, clip_loss_fn):
    """计算CLIPLoss"""
    loss = clip_loss_fn(
        original_img, seq_src,
        pred_img, seq_tgt
    )
    return loss

def train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, batch_size):
    model_G.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_loss_fn = clip_loss.CLIPLoss(device=device, lambda_direction=1.0, lambda_global=0.8)
    clip_loss_total = utils.Averager()

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        src_imgs = batch['src_img']
        tgt_imgs = batch['tgt_img']
        seq_src = batch['seq_src']
        seq_tgt = batch['seq_tgt']
        print('src_img.shape:',src_imgs.shape) # ([2, 144, 192, 192])
        print('seq_src.shape:',seq_src.shape) # torch.Size([4, 1, 768])

        # 对图像进行归一化或去异常值处理
        src_imgs = utils.percentile_clip(src_imgs)
        tgt_imgs = utils.percentile_clip(tgt_imgs)
        print(f"src_imgs_percentile_clip: {src_imgs.shape}") # src_imgs_percentile_clip: torch.Size([2, 144, 192, 192])
        print(f"tgt_imgs_percentile_clip: {tgt_imgs.shape}")

        coord_size = [32, 32, 32]
        coord_hr = utils.make_coord(coord_size, flatten=True)
        coord_hr = coord_hr.clone().detach().cuda().float().unsqueeze(0)

        optimizer_G.zero_grad()
        # 通过模型进行预测
        crop_size = (32, 32, 32)
        pred_imgs = _get_pred(crop_size, overlap_ratio, model_G, src_imgs, tgt_imgs, coord_size, coord_hr, seq_src, seq_tgt)
        print('pred_imgs.shape:', pred_imgs.shape)

        # src_imgs = batch['src_img'].cuda().float()
        # tgt_imgs = batch['tgt_img'].cuda().float()
        # seq_src = batch['seq_src'].cuda().float()
        # seq_tgt = batch['seq_tgt'].cuda().float()
        # print('img.shape:', src_imgs.shape) # ([2, 144, 192, 192])

        # optimizer_G.zero_grad()

        # # 初始化存储预测图像和源/目标图像的列表
        # all_src_patches = []
        # all_tgt_patches = []
        # all_pre_patches = []
        # all_positions = []
        # all_src_imgs = []
        # all_pre_imgs = []
        #
        # # for i in range(src_imgs.size(0)):  # 遍历每个样本
        # #     src_img = src_imgs[i]  # ([144, 192, 192])
        # #     tgt_img = tgt_imgs[i]  # [D, H, W]
        # #     src_seq = seq_src[i]   # [1, 768]
        # #     tgt_seq = seq_tgt[i]   # [1, seq_len]
        # #     print('img.shape:', src_img.shape)
        # #     print('seq.shape:', src_seq.shape)
        # #
        # #     src_seq_batch = src_seq.repeat(batch_size, 1)  # ([2, 768])
        # #     tgt_seq_batch = tgt_seq.repeat(batch_size, 1)  # [batch_size, seq_len]
        # #     print('seq_batch.shape:', src_seq_batch.shape)
        #
        # # 切分图像为 patch
        # src_patches, positions = slice_image(src_imgs.cpu().numpy(), patch_size, overlap_ratio)
        # tgt_patches, _ = slice_image(tgt_imgs.cpu().numpy(), patch_size, overlap_ratio)
        #
        # # 将 patches 转换为 [num_patches, 32, 32, 32]
        # src_patches = src_patches.cuda()  # [num_patches, 32, 32, 32]
        # tgt_patches = tgt_patches.cuda()  # [num_patches, 32, 32, 32]
        #
        # num_patches = src_patches.size(0)
        #
        # # 收集输出结果
        # pre_patches_list = []
        #
        # # 处理每个patch的预测
        # for start_idx in range(0, num_patches, batch_size):
        #     end_idx = min(start_idx + batch_size, num_patches)
        #
        #     # 获取当前批次的 patches
        #     src_patches_batch = src_patches[start_idx:end_idx]  # [batch_size, 32, 32, 32]
        #     tgt_patches_batch = tgt_patches[start_idx:end_idx]  # [batch_size, 32, 32, 32]
        #     print('img_patches_batch.shape:', src_patches_batch.shape)
        #
        #     pre_patches_batch = model_G(src_patches_batch, tgt_patches_batch, src_seq_batch, tgt_seq_batch)
        #
        #     if isinstance(pre_patches_batch, tuple):
        #         pre_patches_batch = pre_patches_batch[0]
        #
        #     pre_patches_list.append(pre_patches_batch.detach().cpu())
        #
        #     pre_patches = torch.cat(pre_patches_list, dim=0).cuda()
        #
        #     # 将所有的 patches 拼接成完整图像
        #     pre_img = reconstruct_image(pre_patches, positions, src_img.shape, patch_size, overlap_ratio)
        #     print('pre_img.shape:', pre_img.shape) # ([144, 192, 192])
        #     print('text.shape:', src_seq.shape) # ([1, 768])
        #
        #     # 收集当前样本的图像和预测结果
        #     all_src_imgs.append(src_img)
        #     all_pre_imgs.append(pre_img)
        #
        #     # 将每个样本的预测拼接到总结果中
        #     all_src_patches.append(src_patches)
        #     all_tgt_patches.append(tgt_patches)
        #     all_pre_patches.append(pre_patches)
        #     all_positions.append(positions)
        #
        # # 将收集到的全批次的源图像和预测图像传入 compute_clip_loss 进行计算
        # all_src_imgs = torch.stack(all_src_imgs).cuda()  # [batch_size, D, H, W]
        # all_pre_imgs = torch.stack(all_pre_imgs).cuda()  # [batch_size, D, H, W]
        #
        # all_src_imgs = all_src_imgs.unsqueeze(1)  # 在第1个维度添加通道维度
        # all_pre_imgs = all_pre_imgs.unsqueeze(1)
        # print('all_imgs.shape:', all_src_imgs.shape) # ([2, 1, 144, 192, 192])
        # # seq_src = seq_src.unsqueeze(0).repeat(2,1)  # [batch_size, seq_len]
        # # seq_tgt = seq_tgt.unsqueeze(0).repeat(2,1)
        # print('all_seq.shape:', seq_src.shape) # ([2, 1, 768])

        # 计算 CLIPLoss
        # loss_clip = compute_clip_loss(all_src_imgs, all_pre_imgs, seq_src, seq_tgt, clip_loss_fn)
        loss_clip = compute_clip_loss(src_imgs, pred_imgs, seq_src, seq_tgt, clip_loss_fn)
        loss_clip.backward()

        # 累加损失
        clip_loss_total.add(loss_clip.item())

        optimizer_G.step()

    return clip_loss_total.item()

def main(config_, save_path):
    global config, log
    config = config_  # config_为config.yaml文件中的内容
    log, _ = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    patch_size = config['training_params']['patch_size']
    overlap_ratio = config['training_params']['overlap_ratio']
    # patch_batch_size = config['training_params']['patch_batch_size']

    train_loader, val_loader = make_data_loaders()
    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    max_val_v = -1e18
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        optimizer_G.param_groups[0]['lr'] = 0.00001
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        log_info.append('lr_G={:.6f}'.format(optimizer_G.param_groups[0]['lr']))

        train_loss = train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, batch_size)
        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        log_info.append('clip_loss={:.4f}'.format(train_loss))

        model_G_ = model_G.module if isinstance(model_G, nn.parallel.DistributedDataParallel) else model_G

        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G_.state_dict()
        optimizer_G_spec = config['optimizer_G']
        optimizer_G_spec['sd_G'] = optimizer_G.state_dict()
        sv_file = {'model_G': model_G_spec, 'optimizer_G': optimizer_G_spec, 'epoch': epoch}

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res = utils.eval_psnr(val_loader, model_G_)
            log_info.append('val: psnr={:.4f}'.format(val_res))

            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

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

    batch_size = config['train_dataset']['batch_size']

    save_name = args.name or '_' + args.config.split('/')[-1][:-len('.yaml')] + '_batch_size_' + str(batch_size)
    if args.tag:
        save_name += '_' + args.tag + '_batch_size_' + str(batch_size)
    print('save_name:', save_name)

    save_path = os.path.join('./save/train/CLIPLoss', save_name)
    os.makedirs(save_path, exist_ok=True)

    main(config, save_path)
