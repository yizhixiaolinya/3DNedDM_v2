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
        sv_file = torch.load(config['resume']) # 加载训练的模型状态(包括模型、优化器、训练的epoch等)
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1 # get epoch
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

        for _ in range(epoch_start - 1):
            lr_scheduler_G.step() # update lr
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


def train(train_loader, model_G, optimizer_G):
    model_G.train()
    # loss_fn = nn.L1Loss()

    # 初始化 CLIPLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_loss_fn = clip_loss.CLIPLoss(device=device, lambda_direction=1.0, lambda_global=0.8)

    loss_0 = utils.Averager()
    loss_1 = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda().float()
        # 3D 图像输入和文本特征的提取
        seq_src = batch['seq_src'].cuda()
        seq_tgt = batch['seq_tgt'].cuda()
        tgt_hr = batch['tgt_hr'].cuda()
        src_hr = batch['src_hr'].cuda()
        # print('frc_hr.shape', src_hr.shape) # [2, 1, 768]
        # print('tgt_hr.shape', tgt_hr.shape)
        # print('seq_src.shape', seq_src.shape) # [2, 32, 32, 32]
        # print('seq_tgt.shape', seq_tgt.shape)
        # 前向传播得到生成的源-目标、目标-源图像对
        pre_src_tgt, pre_tgt_src = model_G(src_hr, tgt_hr, seq_src, seq_tgt)
        print('pre_src_tgt.shape:', pre_src_tgt.shape) # ([2, 1, 32, 32, 32])
        print('pre_tgt_src.shape:', pre_tgt_src.shape)

        # loss_src = loss_fn(pre_src_tgt, tgt_hr)
        # loss_tgt = loss_fn(pre_tgt_src, src_hr)

        # 调用 CLIPLoss 计算损失，使用 seq_src 和 seq_tgt 作为文本输入，tgt_hr 和 src_hr 作为图像输入
        loss_src = clip_loss_fn(src_img=pre_src_tgt, source_class=seq_src, target_img=tgt_hr, target_class=seq_tgt)
        loss_tgt = clip_loss_fn(src_img=pre_tgt_src, source_class=seq_tgt, target_img=src_hr, target_class=seq_src)

        loss_G = loss_src * 0.5 + loss_tgt * 0.5
        loss_0.add(loss_src.item()) # 对应于模型生成的源目标预测与真实目标的损失（loss_src），即源图像经过模型生成后与目标图像的损失
        loss_1.add(loss_tgt.item()) # 对应于模型生成的目标源预测与真实源图像的损失（loss_tgt），即目标图像经过模型生成后与源图像的损失

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    return loss_0.item(), loss_1.item()


def main(config_, save_path):
    global config, log
    config = config_ # config_为config.yaml文件中的内容
    log, _ = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training()
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model_G.cuda()
        model_G = nn.parallel.DistributedDataParallel(model_G)
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

        train_loss = train(train_loader, model_G, optimizer_G)
        if lr_scheduler_G is not None:
           lr_scheduler_G.step()
        log_info.append('loss0={:.4f}'.format(train_loss[0]))
        log_info.append('loss1={:.4f}'.format(train_loss[1]))

        if n_gpus > 1:
            model_G_ = model_G.module
        else:
            model_G_ = model_G

        # Check model_G_ parameters
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
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res0, val_res1 = utils.eval_psnr(val_loader, model_G_)

            log_info.append('val: psnr0={:.4f}'.format(val_res0))
            log_info.append('val: psnr1={:.4f}'.format(val_res1))

            if val_res0 + val_res1 > max_val_v:
                max_val_v = val_res0 + val_res1
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

    # 设置GPU
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

    # 设置保存路径：存储到 `train/batch_size_<batch_size>` 目录
    save_path = os.path.join('./save/train/CLIPLoss', save_name)

    # 确保路径存在
    os.makedirs(save_path, exist_ok=True)

    # 开始训练
    main(config, save_path)
