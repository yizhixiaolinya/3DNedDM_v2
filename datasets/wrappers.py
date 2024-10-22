import random
from torch.utils.data import Dataset
from datasets import register
import numpy as np
import utils

@register('sr-implicit-paired')  # 使用register装饰器注册类名为'sr-implicit-paired'
class SRImplicitPaired(Dataset):  # 定义SRImplicitPaired类，继承自Dataset

    def __init__(self, dataset, scale_min=1, scale_max=None, augment=False, sample_q=None):  # 初始化方法
        self.dataset = dataset  # 保存传入的数据集
        self.scale_min = scale_min  # 保存最小缩放比例
        self.scale_max = scale_max  # 保存最大缩放比例
        self.augment = augment  # 保存是否进行数据增强
        self.sample_q = sample_q  # 保存采样参数

    def __len__(self):  # 定义获取数据集长度的方法
        return len(self.dataset)  # 返回数据集的长度

    def __getitem__(self, idx):  # 定义获取数据项的方法
        patch_src_hr, patch_tgt_hr, seq_src, seq_tgt = self.dataset[idx]  # 从数据集中获取源和目标图像块及其对应的序列
        patch_src_hr = utils.percentile_clip(patch_src_hr)  # 对源图像块进行百分位剪裁
        patch_tgt_hr = utils.percentile_clip(patch_tgt_hr)  # 对目标图像块进行百分位剪裁
        non_zero = np.nonzero(patch_src_hr)  # 获取源图像块中非零元素的索引
        min_indice = np.min(non_zero, axis=1)  # 获取非零元素的最小索引
        max_indice = np.max(non_zero, axis=1)  # 获取非零元素的最大索引
        patch_src_hr = patch_src_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]  # 裁剪源图像块
        patch_tgt_hr = patch_tgt_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]  # 裁剪目标图像块
        size = 32  # 定义裁剪块的大小
        h0 = random.randint(0, patch_src_hr.shape[0] - size)  # 随机生成裁剪块的起始高度
        w0 = random.randint(0, patch_src_hr.shape[1] - size)  # 随机生成裁剪块的起始宽度
        d0 = random.randint(0, patch_src_hr.shape[2] - size)  # 随机生成裁剪块的起始深度
        patch_src_hr = patch_src_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size]  # 裁剪源图像块
        patch_tgt_hr = patch_tgt_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size]  # 裁剪目标图像块

        return {  # 返回包含源和目标图像块及其对应序列的字典
            'src_hr': patch_src_hr,
            'tgt_hr': patch_tgt_hr,
            'seq_src': seq_src,
            'seq_tgt': seq_tgt
        }


