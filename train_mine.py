# train
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

import numpy as np
import torch.nn.functional as F
import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import random

def make_data_loader(spec, epoch, epoch_threshold, tag=''):
    '''Create data loader'''
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])

    # 根据 epoch 动态选择 wrapper
    if epoch <= epoch_threshold:
        # 使用 wrapper_paired
        dataset = datasets.make(spec['wrapper_paired'], args={'dataset': dataset})
    else:
        # 使用 wrapper_full
        dataset = datasets.make(spec['wrapper_full'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=4, pin_memory=True)
    return loader

def make_data_loaders(epoch, epoch_threshold):
    '''Make data loaders'''
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', epoch=epoch, epoch_threshold=epoch_threshold)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', epoch=epoch, epoch_threshold=epoch_threshold)
    return train_loader, val_loader

def prepare_training():
    '''Prepare training'''
    if config.get('resume') is not None:
        # 加载最近的训练模型状态
        sv_file = torch.load(config['resume'])  # 加载模型、优化器、训练的epoch等状态
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()

        # 加载优化器状态
        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)

        # optimizer_G = utils.make_optimizer_G(
        #     model_G.parameters(), config['optimizer_G'])

        # epoch_start = sv_file['epoch'] + 1  # 继续从上次的 epoch 开始
        epoch_start = 0
        
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


def train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, writer, epoch, epoch_threshold):
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
        if epoch <= epoch_threshold:
            # 利用之前的代码进行L1 loss的训练
            for k, v in batch.items():
                batch[k] = v.cuda().float()
            seq_src = batch['seq_src'].cuda()
            seq_tgt = batch['seq_tgt'].cuda()
            tgt_hr = batch['tgt_hr'].cuda()
            src_hr = batch['src_hr'].cuda()
            # tgt_hr = utils.crop_img(tgt_hr, 96).cuda()
            # src_hr = utils.crop_img(src_hr, 96).cuda()

            pre_src_tgt, pre_tgt_src = model_G(src_hr, tgt_hr, seq_src, seq_tgt)

            loss_src = loss_fn(pre_src_tgt, tgt_hr)
            loss_tgt = loss_fn(pre_tgt_src, src_hr)

            loss_total = loss_src * 0.5 + loss_tgt * 0.5
            loss_0.add(loss_src.item())  # 对应于模型生成的源目标预测与真实目标的损失（loss_src），即源图像经过模型生成后与目标图像的损失
            loss_1.add(loss_tgt.item())  # 对应于模型生成的目标源预测与真实源图像的损失（loss_tgt），即目标图像经过模型生成后与源图像的损失
            writer.add_scalar('Loss/loss0', loss_0.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/loss1', loss_1.item(), epoch * len(train_loader) + batch_idx)


            optimizer_G.zero_grad()
            loss_total.backward()
            optimizer_G.step()

        else:

            src_imgs = batch['src_img']
            tgt_imgs = batch['tgt_img']
            src_imgs = utils.crop_img(src_imgs,96)
            tgt_imgs = utils.crop_img(tgt_imgs, 96)
            src_seq = batch['seq_src'].cuda().float()
            tgt_seq = batch['seq_tgt'].cuda().float()

            optimizer_G.zero_grad()

            # 切分图像为 patch
            src_patches, positions = utils.slice_image(src_imgs.cpu().numpy(), patch_size, overlap_ratio)
            tgt_patches, _ = utils.slice_image(tgt_imgs.cpu().numpy(), patch_size, overlap_ratio)

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
            pre_img_src2tgt = utils.reconstruct_image(pre_patches_src2tgt, positions, src_imgs.shape, patch_size, overlap_ratio)
            pre_img_tgt2src = utils.reconstruct_image(pre_patches_tgt2src, positions, tgt_imgs.shape, patch_size, overlap_ratio)
            # print('pre_img.shape:', pre_img_src2tgt.shape) # torch.Size([2, 96, 96, 96])
            # print('text.shape:', src_seq.shape) # torch.Size([2, 1, 768])

            cropped_src, cropped_tgt, cropped_src2tgt, cropped_tgt2src = utils.get_non_zero_random_crop(src_imgs.cpu(),
                                                                                                  tgt_imgs.cpu(),
                                                                                                  pre_img_src2tgt.cpu(),
                                                                                                  pre_img_tgt2src.cpu(),
                                                                                                  patch_size)

            cropped_src = cropped_src2tgt.unsqueeze(1).cuda().float()
            cropped_tgt = cropped_tgt.unsqueeze(1).cuda().float()
            cropped_src2tgt = cropped_src2tgt.unsqueeze(1).cuda().float()
            cropped_tgt2src = cropped_tgt2src.unsqueeze(1).cuda().float()

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
    epoch_threshold = config.get('epoch_threshold', 30) 
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
    # print('SAVED PATH',flush=True)
    # 初始化 TensorBoard
    # writer 已经在 set_save_path 中初始化

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    patch_size = config['training_params']['patch_size']
    overlap_ratio = config['training_params']['overlap_ratio']

    train_loader, val_loader = make_data_loaders(epoch=1, epoch_threshold=epoch_threshold)

    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training()

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

        # 当epoch > threshold + 1 时
        if epoch == epoch_threshold + 1:
            train_loader, val_loader = make_data_loaders(epoch=epoch, epoch_threshold=epoch_threshold)
        # 传递 writer
        train_loss = train(train_loader, model_G, optimizer_G, patch_size, overlap_ratio, writer, epoch, epoch_threshold)

        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        log_info.append('total_loss={:.4f}'.format(train_loss))

        

        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G.state_dict()
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
            val_res_src2tgt, val_res_tgt2src = utils.eval_psnr(val_loader, model_G, epoch, patch_size, overlap_ratio, epoch_threshold)
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

    save_path = os.path.join('/home_data/home/linxin2024/code/3DMedDM_v2/save/train/', save_name)
    save_path = os.path.join(save_path,datetime.now().strftime("%Y%m%d_%H%M%S"))

    os.makedirs(save_path, exist_ok=True)

    main(config, save_path)
