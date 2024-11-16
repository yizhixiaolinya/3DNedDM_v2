import os
import time
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from itertools import product

# add clip_loss
import yaml
from CLIP.model import CLIP # 2D
from utils_clip import load_config_file
import torchvision.transforms as transforms

def eval_psnr(loader, model, epoch, patch_size, overlap_ratio, epoch_threshold):
    """评估模型信噪比"""
    model.eval()

    # 初始化度量函数和结果累加器
    metric_fn = calc_psnr
    val_res1 = Averager()
    val_res0 = Averager()

    with torch.no_grad():  # 将无梯度计算移到循环外
        for batch in tqdm(loader, leave=False, desc='val'):
            for k, v in batch.items():
                batch[k] = v.cuda().float()

            seq_src = batch['seq_src']
            seq_tgt = batch['seq_tgt']

            if epoch <= epoch_threshold:
                tgt_hr = batch['tgt_hr']
                src_hr = batch['src_hr']
                pre_src_tgt, pre_tgt_src = model(src_hr, tgt_hr, seq_src, seq_tgt)

            else:
                tgt_hr = batch['tgt_img']
                src_hr = batch['src_img']
    
                # 切分图像为 patch
                src_patches, positions = slice_image(src_hr.cpu().numpy(), patch_size, overlap_ratio)
                tgt_patches, _ = slice_image(tgt_hr.cpu().numpy(), patch_size, overlap_ratio)

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

                    seq_src = seq_src.reshape(seq_src.shape[0], 1, seq_src.shape[-1])
                    seq_tgt = seq_tgt.reshape(seq_tgt.shape[0], 1, seq_tgt.shape[-1])
                    pre_patches_batch_src2tgt, pre_patches_batch_tgt2src = model(src_patches_batch.cuda(), tgt_patches_batch.cuda(), seq_src, seq_tgt)

                    if isinstance(pre_patches_batch_src2tgt, tuple):
                        pre_patches_batch_src2tgt = pre_patches_batch_src2tgt[0]

                    if isinstance(pre_patches_batch_tgt2src, tuple):
                        pre_patches_batch_tgt2src = pre_patches_batch_tgt2src[0]

                    pre_patches_src2tgt_list.append(pre_patches_batch_src2tgt.cpu())
                    pre_patches_tgt2src_list.append(pre_patches_batch_tgt2src.cpu())

                pre_patches_src2tgt = torch.cat(pre_patches_src2tgt_list, dim=0).cpu()
                pre_patches_tgt2src = torch.cat(pre_patches_tgt2src_list, dim=0).cpu()

                # 将所有的 patches 拼接成完整图像
                pre_src_tgt = reconstruct_image(pre_patches_src2tgt, positions, src_hr.shape, patch_size, overlap_ratio)
                pre_tgt_src = reconstruct_image(pre_patches_tgt2src, positions, tgt_hr.shape, patch_size, overlap_ratio)

            res0 = metric_fn(pre_src_tgt.cuda(), tgt_hr.cuda())
            res1 = metric_fn(pre_tgt_src.cuda(), src_hr.cuda())

            val_res0.add(res0.item(), src_hr.shape[0])
            val_res1.add(res1.item(), src_hr.shape[0])

    return val_res0.item(), val_res1.item()

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


def percentile_clip(input_tensor, reference_tensor=None, p_min=0.01, p_max=99.9, strictlyPositive=True):
    """
    The percentile_clip function clips the values of an input tensor to specified percentiles and normalizes them to the range [0, 1]. This is useful for preprocessing data.
    :param input_tensor: The input tensor to be clipped and normalized.
    """
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile
    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
   
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]
    # print(output_tensor.min(),output_tensor.max())
    return output_tensor

def random_selection(input_list):
    num_to_select = random.randint(1, 3)  # 随机选择1到3个数字
    selected_numbers = random.sample(input_list, num_to_select)  # 从列表中选择随机数字
    return selected_numbers

class Loss_CC(torch.nn.Module):
    """
    The Loss_CC class defines a custom loss function that computes the correlation coefficient loss between feature maps.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, m):
        b, c, h, w = m.shape
        m = m.reshape(b, c, h*w)
        m = torch.nn.functional.normalize(m, dim=2, p=2)
        m_T = torch.transpose(m, 1, 2)
        m_cc = torch.matmul(m, m_T)
        mask = torch.eye(c).unsqueeze(0).repeat(b,1,1).cuda()
        m_cc = m_cc.masked_fill(mask==1, 0)
        loss = torch.sum(m_cc**2)/(b*c*(c-1))
        return loss
    
class Averager():
    """
    The Averager class is a utility to compute the running average of values, which is useful for tracking metrics during training or evaluation
    """
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True, ask_user=False):
    if os.path.exists(path):
        if remove:
            if ask_user:
                try:
                    response = input(f"{path} exists, remove? (y/[n]): ")
                    if response.lower() != 'y':
                        raise FileExistsError(f"Path '{path}' already exists and was not removed.")
                except EOFError:
                    # 如果在非交互式环境中运行，自动选择不删除
                    raise FileExistsError(f"Path '{path}' already exists and was not removed (EOFError).")
            # 如果选择删除或没有启用用户询问，删除路径
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove=True, ask_user=False):
    ensure_path(save_path, remove=remove, ask_user=ask_user)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def make_optimizer_G(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_G'])
    return optimizer

def make_optimizer_D(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_D'])
    return optimizer

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def calc_psnr(sr, hr):
    diff = (sr - hr) 
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse)

def write_middle_feature(intermediate_output):
    for i in range(intermediate_output.shape[1]):
        activation = intermediate_output[0, i, :, :, :]
        plt.savefig(f'./save/layer_{i}_activation_{activation}.png')  # Save each activation as a PNG file
        plt.clf()

def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)

class GANLoss(nn.Module):
    """
    The GANLoss class defines a loss function for Generative Adversarial Networks (GANs), supporting both least-squares and binary cross-entropy loss.
    """
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
