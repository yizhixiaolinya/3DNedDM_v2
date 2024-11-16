# train
import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
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
    '''创建数据加载器'''
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
    '''创建训练和验证的数据加载器'''
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', epoch=epoch, epoch_threshold=epoch_threshold)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', epoch=epoch, epoch_threshold=epoch_threshold)
    return train_loader, val_loader


def prepare_training():
    '''准备训练过程'''
    if config.get('resume') is not None:
        # 加载最近的训练模型状态
        sv_file = torch.load(config['resume'])  # 加载模型、优化器、训练的epoch等状态
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()
        model_D = models.make(sv_file['model_D'], load_sd=True).cuda()
        
        # 加载优化器状态
        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)
        optimizer_D = utils.make_optimizer_D(
            model_D.parameters(), sv_file['optimizer_D'], load_sd=True)
        
        # 如果要重新加载学习率
        new_lr = config['optimizer_G']['args'].get('lr', 5e-4)  # 从配置文件中获取新的学习率
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = new_lr
        
        epoch_start = sv_file['epoch'] + 1  # 继续从上次的 epoch 开始
        
    else:
        # 从头开始训练
        model_G = models.make(config['model_G']).cuda()
        # config['model_D']['args']['in_dim'] = config['training_params']['patch_size'][0] * config['training_params']['patch_size'][1] * config['training_params']['patch_size'][2]
        model_D = models.make(config['model_D']).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), config['optimizer_G']
        )
        optimizer_D = utils.make_optimizer_D(
            model_D.parameters(), config['optimizer_D']
        )

        epoch_start = 1

    # 初始化自动调度器或多步调度器
    if config.get('use_auto_lr_scheduler', False):
        lr_scheduler_G = ReduceLROnPlateau(
            optimizer_G,
            mode=config['lr_scheduler'].get('mode', 'min'),  # 如果使用 PSNR，则应设为 'max'
            factor=config['lr_scheduler'].get('factor', 0.5),
            patience=config['lr_scheduler'].get('patience', 1),
            threshold=config['lr_scheduler'].get('threshold', 0.1),
            cooldown=config['lr_scheduler'].get('cooldown', 1),
            min_lr=config['lr_scheduler'].get('min_lr', 1e-6)
        )
    elif config.get('multi_step_lr') is not None:
        lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])
    else:
        lr_scheduler_G = None

    # 记录模型参数数量
    log('model_G: #params={}'.format(utils.compute_num_params(model_G, text=True)))
    log('model_D: #params={}'.format(utils.compute_num_params(model_D, text=True)))
 
    return model_G, model_D, optimizer_G, optimizer_D, epoch_start, lr_scheduler_G


def train(train_loader, model_G, model_D, optimizer_G, optimizer_D, patch_size, overlap_ratio, writer, epoch, epoch_threshold):
    '''训练一个 epoch'''
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_G.train()
    model_D.train()

    loss_fn = nn.L1Loss()
    clip_loss_fn = clip_loss.CLIPLoss(device=device, lambda_direction=1.0, lambda_global=0.8)

    # 使用标准GAN的损失函数
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)

    # loss_0 = utils.Averager()
    # loss_1 = utils.Averager()
    clip_loss_s2t = utils.Averager()
    clip_loss_t2s = utils.Averager()

    # 累积每个 batch 的 total loss
    epoch_loss_total = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, leave=True, desc='train')):
        if epoch <= epoch_threshold:
            # 使用局部损失（不包含clip_loss）
            for k, v in batch.items():
                batch[k] = v.cuda().float()
            seq_src = batch['seq_src']
            seq_tgt = batch['seq_tgt']
            tgt_hr = batch['tgt_hr']
            src_hr = batch['src_hr']
            
            src_imgs, tgt_imgs, src_seq, tgt_seq = src_hr, tgt_hr, seq_src, seq_tgt

        else:
            # 使用局部+全局损失（包含clip_loss）
            src_imgs = batch['src_img']
            tgt_imgs = batch['tgt_img']
            src_imgs = utils.crop_img(src_imgs, 96)
            tgt_imgs = utils.crop_img(tgt_imgs, 96)
            src_seq = batch['seq_src'].cuda().float()
            tgt_seq = batch['seq_tgt'].cuda().float()

            # 切分图像为 patch
            src_patches, positions = utils.slice_image(src_imgs.cpu().numpy(), patch_size, overlap_ratio)
            tgt_patches, _ = utils.slice_image(tgt_imgs.cpu().numpy(), patch_size, overlap_ratio)

            num_patches = src_patches.size(0)
            pre_patches_src2tgt_list = []
            pre_patches_tgt2src_list = []

            # 处理每个patch的预测
            for start_idx in range(num_patches):
                src_patches_batch = src_patches[start_idx]
                tgt_patches_batch = tgt_patches[start_idx]

                src_seq_reshaped = src_seq.view(src_seq.size(0), 1, src_seq.size(-1))
                tgt_seq_reshaped = tgt_seq.view(tgt_seq.size(0), 1, tgt_seq.size(-1))
                pre_patches_batch_src2tgt, pre_patches_batch_tgt2src = model_G(
                    src_patches_batch.cuda(), tgt_patches_batch.cuda(), src_seq_reshaped, tgt_seq_reshaped
                )

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

            # 计算clip_loss
            loss_clip_src2tgt = clip_loss_fn(src_imgs, src_seq, pre_img_src2tgt, tgt_seq)
            loss_clip_tgt2src = clip_loss_fn(tgt_imgs, tgt_seq, pre_img_tgt2src, src_seq)

        ############################
        # 训练判别器
        ############################
        optimizer_D.zero_grad()

        # 使用生成器生成假图像
        fake_tgt, fake_src = model_G(src_imgs, tgt_imgs, src_seq, tgt_seq)
        print(src_imgs.shape, tgt_imgs.shape, fake_src.shape, fake_tgt.shape)
        # torch.Size([2, 32, 32, 32]) torch.Size([2, 32, 32, 32]) torch.Size([2, 1, 32, 32, 32]) torch.Size([2, 1, 32, 32, 32])
        
        # 5D
        src_imgs = torch.unsqueeze(src_imgs, dim=1)
        tgt_imgs = torch.unsqueeze(tgt_imgs, dim=1)
        # print(src_imgs.shape, tgt_imgs.shape)
        # print('5D done')

        # 判别器对真实图像的输出
        real_validity = model_D(src_imgs, fake_src.detach())
        # 判别器对生成图像的输出
        fake_validity = model_D(tgt_imgs, fake_tgt.detach())

        # 创建真实标签和假标签
        valid = torch.ones_like(fake_validity).to(device)
        fake = torch.zeros_like(fake_validity).to(device)

        # 判别器损失
        d_loss_real = criterion_GAN((real_validity), valid)
        d_loss_fake = criterion_GAN((fake_validity), fake)
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_loss.backward()
        optimizer_D.step()

        ############################
        # 训练生成器
        ############################
        optimizer_G.zero_grad()

        # 4D
        src_imgs = torch.squeeze(src_imgs, dim=1)
        tgt_imgs = torch.squeeze(tgt_imgs, dim=1)
        # print(src_imgs.shape, tgt_imgs.shape)
        # print('4D Done')

        # 重新生成假图像（确保梯度可以传回生成器）
        fake_tgt, fake_src = model_G(src_imgs, tgt_imgs, src_seq, tgt_seq)
        fake_validity = model_D(fake_src, fake_tgt)

        # 生成器对抗损失
        g_loss_adv = criterion_GAN(fake_validity, valid)

        # 像素级一致性损失
        g_loss_pixel = loss_fn(fake_src, src_imgs) + loss_fn(fake_tgt, tgt_imgs)

        # 根据 epoch_threshold 判断是否加入 clip_loss
        if epoch > epoch_threshold:
            g_loss = g_loss_adv + g_loss_pixel + (loss_clip_src2tgt + loss_clip_tgt2src)
            clip_loss_s2t.add(loss_clip_src2tgt.item())
            clip_loss_t2s.add(loss_clip_tgt2src.item())
            writer.add_scalar('Loss/cliploss_src2tgt', clip_loss_s2t.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/cliploss_tgt2src', clip_loss_t2s.item(), epoch * len(train_loader) + batch_idx)
        else:
            g_loss = g_loss_adv + g_loss_pixel

        g_loss.backward()
        optimizer_G.step()

        # 记录损失
        writer.add_scalar('Loss/D_real', d_loss_real.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/D_fake', d_loss_fake.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/D', d_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/G_adv', g_loss_adv.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/G_pixel', g_loss_pixel.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/G', g_loss.item(), epoch * len(train_loader) + batch_idx)

        

        epoch_loss_total += g_loss.item()

    return epoch_loss_total


def main(config_, save_path):
    global config, log
    config = config_  # config_为config.yaml文件中的内容

    # 从配置中获取是否询问用户
    ask_user = config.get('ask_user', False)
    epoch_threshold = config.get('epoch_threshold', 30)

    # 当 resume 不为空时，不删除已有路径，保留日志和其他文件
    if config.get('resume') is not None:
        remove_save_path = False
    else:
        # 如果不从检查点继续训练，可以根据需要决定是否删除
        remove_save_path = config.get('remove_save_path', True)  # 从配置中获取，默认删除

    # 设置保存路径，传递 remove 和 ask_user 参数
    log, writer = utils.set_save_path(save_path, remove=remove_save_path, ask_user=ask_user)

    # 保存配置文件到保存路径
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    patch_size = config['training_params']['patch_size']
    overlap_ratio = config['training_params']['overlap_ratio']

    # 创建数据加载器
    train_loader, val_loader = make_data_loaders(epoch=1, epoch_threshold=epoch_threshold)

     # 准备训练（加载模型、优化器、调度器等）
    model_G, model_D, optimizer_G, optimizer_D, epoch_start, lr_scheduler_G = prepare_training()

    resumed_epoch = epoch_start  # 继续训练的起始 epoch

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    # 初始化两个最大 PSNR 变量（假设 PSNR 越高越好）
    max_val_v_src2tgt = -1e18
    max_val_v_tgt2src = -1e18

    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()

        log_info = [f'epoch {epoch}/{epoch_max}']

        # 当 epoch > threshold 时，重新创建数据加载器
        if epoch == epoch_threshold + 1:
            train_loader, val_loader = make_data_loaders(epoch=epoch, epoch_threshold=epoch_threshold)

        train_loss = train(train_loader, model_G, model_D, optimizer_G, optimizer_D, patch_size, overlap_ratio, writer, epoch, epoch_threshold)
        
        log_info.append(f'train_loss={train_loss:.4f}')

        # 保存最新的 epoch-last.pth
        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G.state_dict()
        model_D_spec = config['model_D']
        model_D_spec['sd_D'] = model_D.state_dict()
        optimizer_G_spec = config['optimizer_G']
        optimizer_G_spec['sd_G'] = optimizer_G.state_dict()
        optimizer_D_spec = config['optimizer_D']
        optimizer_D_spec['sd_D'] = optimizer_D.state_dict()
        sv_file = {
            'model_G': model_G_spec,
            'model_D': model_D_spec,
            'optimizer_G': optimizer_G_spec,
            'optimizer_D': optimizer_D_spec,
            'epoch': epoch
        }
        
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
            if isinstance(lr_scheduler_G, ReduceLROnPlateau):
                # 计算验证损失（ PSNR 越高越好，因此设置 mode='max'）
                val_loss = (val_res_src2tgt + val_res_tgt2src) / 2.0
                # 更新调度器，传入 val_loss
                lr_scheduler_G.step(val_loss)
                log_info.append(f'val_psnr={val_loss:.4f}')
                current_lr = optimizer_G.param_groups[0]['lr']
                log_info.append(f'Updated lr_G={current_lr:.6f}')
            elif lr_scheduler_G is not None:
                # 对于 MultiStepLR 或其他调度器
                lr_scheduler_G.step()
                
            # 检查并保存最佳 src2tgt 模型
            if val_res_src2tgt > max_val_v_src2tgt:
                max_val_v_src2tgt = val_res_src2tgt
                torch.save(sv_file, os.path.join(save_path, 'epoch-best_src2tgt.pth'))

            # 检查并保存最佳 tgt2src 模型
            if val_res_tgt2src > max_val_v_tgt2src:
                max_val_v_tgt2src = val_res_tgt2src
                torch.save(sv_file, os.path.join(save_path, 'epoch-best_tgt2src.pth'))

        # 记录时间
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        # 打印日志
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
    save_path = os.path.join(save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))

    os.makedirs(save_path, exist_ok=True)

    main(config, save_path)
