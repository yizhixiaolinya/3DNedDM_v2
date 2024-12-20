import models
import torch
import SimpleITK as sitk
from scipy import ndimage as nd
import utils
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import os
from itertools import product
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from CLIP.model import CLIP
from utils_clip import load_config_file
import time
from concurrent.futures import ThreadPoolExecutor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint_path = '/public_bme/home/linxin/13119176/checkpoint_CLIP.pt'
MODEL_CONFIG_PATH = 'CLIP/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

tokenizer = SimpleTokenizer()
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

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
    new_spacing = (ori_spacing[0]/scale0, ori_spacing[1]/scale1, ori_spacing[2]/scale2)
    return new_spacing

def tokenize(texts, tokenizer, context_length=90):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
        
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
    '''
    [batch_size, channels, depth, height, width]
    train中的维度
    seq_src shape: torch.Size([2, 1, 768])
    seq_tgt shape: torch.Size([2, 1, 768])
    tgt_hr shape: torch.Size([2, 32, 32, 32])
    src_hr shape: torch.Size([2, 32, 32, 32])
    pre_src_tgt.shape: torch.Size([2, 1, 32, 32, 32])
    pre_tgt_src.shape: torch.Size([2, 1, 32, 32, 32])
    
    '''
    # print('seq_src.shape:', seq_src.shape) # torch.Size([1, 768])
    # 调整 seq_src 和 seq_tgt 的维度，使其与训练时保持一致
    seq_src = seq_src.unsqueeze(0).repeat(2, 1, 1) # [2, 1, 768]
    seq_tgt = seq_tgt.unsqueeze(0).repeat(2, 1, 1) # [2, 1, 768]
    # print('seq_src.shape:', seq_src.shape)

    W, H, D = img_vol_0.shape # 获取长宽高
    # crop_size 表示每次裁剪的3D图像块的大小
    W_po, H_po, D_po = crop_size[0], crop_size[1], crop_size[2]
    # coord_size 是在高分辨率图像下的目标块尺寸，通常用于将低分辨率图像与高分辨率坐标进行对齐
    W_pt, H_pt, D_pt = coord_size[0], coord_size[1], coord_size[2]
    # 计算缩放比例
    scale0 = W_pt/W_po
    scale1 = H_pt/H_po
    scale2 = D_pt/D_po
    # 计算高分辨率的图像尺寸
    W_t = int(W * scale0)
    H_t = int(H * scale1)
    D_t = int(D * scale2)
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    # 生成预测矩阵
    pred_0_1 = np.zeros((W_t, H_t, D_t))
    pred_1_0 = np.zeros((W_t, H_t, D_t))
    freq_rec = np.zeros((W_t, H_t, D_t)) # 用于记录每个位置预测的次数，方便在最终进行平均
    start_time = time.time()

    for start_pos in pos:
        # 通过分割图像为小块（patch），然后对每个小块进行预测，最后将所有小块的预测结果组合回完整的高分辨率图像中
        # 提取低分辨率图像块
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], start_pos[2]:start_pos[2] + crop_size[2]]
        img_1_lr_patch = img_vol_1[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], start_pos[2]:start_pos[2] + crop_size[2]]
        img_0_lr_patch = torch.tensor(img_0_lr_patch).cuda().float().unsqueeze(0).repeat(2, 1, 1, 1) # unsqueeze(0) 两次的作用是扩展图像块的维度，使其从 [depth, height, width] 转换为 [batch_size, channels, depth, height, width] 的形状
        img_1_lr_patch = torch.tensor(img_1_lr_patch).cuda().float().unsqueeze(0).repeat(2, 1, 1, 1) 
        # print('img_0_lr_patch.shape:', img_0_lr_patch.shape) # [2, 20, 60, 60] ->(crop_size修改后) [2, 8, 32, 32]
        # print('img_1_lr_patch.shape:', img_1_lr_patch.shape)

        model.eval()
        with torch.no_grad():
            # 将两个图像块 img_0_lr_patch 和 img_1_lr_patch 以及相应的文本嵌入 seq_src 和 seq_tgt 传入模型进行预测，生成预测结果 pred_0_1_patch
            pred_0_1_patch = model(img_0_lr_patch, img_1_lr_patch, seq_src.cuda().float(), seq_tgt.cuda().float())
            print('img_0_lr_patch.shape',img_0_lr_patch.shape)
            print('seq_src.shape',seq_src.shape)

            # 查看tuple的结构和每个元素的内容
            for idx, item in enumerate(pred_0_1_patch):
                print(f"Element {idx}: Type: {type(item)}, Shape: {item.shape if isinstance(item, torch.Tensor) else 'N/A'}")
                # 见lccd.py 中的 tgt_out 和 src_out, Shape: torch.Size([2, 1, 8, 32, 32])

            # 获取主输出
            pred_0_1_patch = pred_0_1_patch[0]

        # 使用 reshape(W_pt, H_pt, D_pt) 将预测结果重新塑形为高分辨率图像块的尺寸
        # pred_0_1_patch = pred_0_1_patch.squeeze(0).cpu().numpy().reshape(W_pt, H_pt, D_pt)
        # print('pred_0_1_patch:', pred_0_1_patch)

        print(f"Original pred_0_1_patch shape before squeezing: {pred_0_1_patch.shape}")

        # 如果 batch_size 为 2，需要逐个处理每个 batch
        # 去掉 channel 维度
        pred_0_1_patch = pred_0_1_patch.squeeze(1).cpu().numpy()  # 去掉 channel 维度

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
        pred_0_1[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt, target_pos2:target_pos2 + D_pt] += single_patch
            
        # 记录每个位置的预测次数
        freq_rec[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt, target_pos2:target_pos2 + D_pt] += 1
    
    end_time = time.time()
    print(end_time-start_time)

    # 计算预测的最终输出
    pred_0_1_img = pred_0_1 / freq_rec
    # print('pred_0_1_img:',pred_0_1_img)

    return pred_0_1_img

def psnr(ref,ret):
    err = ref - ret
    return -10*np.log10(np.mean(err**2))


psnr_0_1_list = []
psnr_1_0_list = []
ssim_0_1_list = []
ssim_1_0_list = []
model_pth = '/public_bme/data/linxin_debug/Loss/for_debug_pth/_train_lccd_sr_batch_size_2/20241027_131732/epoch-best_src2tgt.pth'
model_img = models.make(torch.load(model_pth)['model_G'], load_sd=True).cuda()
# print(model_img) -> model_img.out

# 读取T1和SWI影像路径文件
t1_img_file = r'/home_data/home/linxin2024/code/3DMedDM_v2/data_try_lx/demo/demo_T1_img.txt'
swi_img_file = r'/home_data/home/linxin2024/code/3DMedDM_v2/data_try_lx/demo/demo_SWI_img.txt'

# 读取影像路径列表
with open(t1_img_file, 'r') as f_t1, open(swi_img_file, 'r') as f_swi:
    t1_img_paths = [line.strip() for line in f_t1.readlines()]
    swi_img_paths = [line.strip() for line in f_swi.readlines()]

# 读取Prompt文件
prompt_M1 = r'/home_data/home/linxin2024/code/3DMedDM_v2/data_try_lx/demo/demo_T1_prompt.txt'
prompt_M2 = r'/home_data/home/linxin2024/code/3DMedDM_v2/data_try_lx/demo/demo_SWI_prompt.txt'
with open(prompt_M1) as f1, open(prompt_M2) as f2:
    lines_M1 = f1.readlines()
    lines_M2 = f2.readlines()

# 循环读取每对图像，进行推理
for idx, (img_0_file_path, img_1_file_path) in enumerate(zip(t1_img_paths, swi_img_paths)):
    start_time_0 = time.time()

    # 使用 SimpleITK 读取图像
    img_0 = sitk.ReadImage(img_0_file_path)
    img_0_spacing = img_0.GetSpacing()
    img_vol_0 = sitk.GetArrayFromImage(img_0)

    H, W, D = img_vol_0.shape
    img_vol_0 = img_pad(img_vol_0, target_shape=(H, W, D))

    img_1 = sitk.ReadImage(img_1_file_path)
    img_1_spacing = img_1.GetSpacing()
    img_vol_1 = sitk.GetArrayFromImage(img_1)

    img_vol_1 = img_pad(img_vol_1, target_shape=(H, W, D))

    # 对图像进行归一化或去异常值处理
    img_vol_0 = utils.percentile_clip(img_vol_0)
    img_vol_1 = utils.percentile_clip(img_vol_1)
    print(f"Image {os.path.basename(img_0_file_path)} processed dimensions: {img_vol_0.shape}") # (192, 192, 144)
    print(f"Image {os.path.basename(img_1_file_path)} processed dimensions: {img_vol_1.shape}")

    coord_size = [32, 32, 32]
    coord_hr = utils.make_coord(coord_size, flatten=True)
    coord_hr = coord_hr.clone().detach().cuda().float().unsqueeze(0)

    # 提取对应的文本
    text_src = lines_M1[idx].replace('"', '').strip((lines_M1[idx].strip().split(':'))[0]).strip()
    text_tgt = lines_M2[idx].replace('"', '').strip((lines_M2[idx].strip().split(':'))[0]).strip()

    # 进行文本token化
    seq_src = tokenize(text_src, tokenizer).cuda()
    with torch.no_grad():
        seq_src = model.encode_text(seq_src)
    seq_tgt = tokenize(text_tgt, tokenizer).cuda()
    with torch.no_grad():
        seq_tgt = model.encode_text(seq_tgt)

    # 通过模型进行预测
    crop_size = (32, 32, 32)
    pred_0_1 = _get_pred(crop_size, 0.75, model_img, img_vol_0, img_vol_1, coord_size, coord_hr, seq_src, seq_tgt)

    # 保存从T1到SWI的图像
    new_spacing_1 = set_new_spacing(img_1_spacing, coord_size, crop_size)
    output_file_0_1 = os.path.join('/home_data/home/linxin2024/code/3DMedDM_v2/save/demo/clip_L1_loss_s_t_overlap_0.75', f'T12SWI_{idx}.nii.gz')
    utils.write_img(pred_0_1, output_file_0_1, img_1_file_path, new_spacing=new_spacing_1)

    # 通过模型进行预测，从SWI到T1
    pred_1_0 = _get_pred(crop_size, 0.75, model_img, img_vol_1, img_vol_0, coord_size, coord_hr, seq_tgt, seq_src)

    # 保存从SWI到T1的图像
    new_spacing_0 = set_new_spacing(img_0_spacing, coord_size, crop_size)
    output_file_1_0 = os.path.join('/home_data/home/linxin2024/code/3DMedDM_v2/save/demo/clip_L1_loss_s_t_overlap_0.75', f'SWI2T1_{idx}.nii.gz')
    utils.write_img(pred_1_0, output_file_1_0, img_0_file_path, new_spacing=new_spacing_0)

    # utils.write_img(pred_1_0, os.path.join('/public/home/v-wangyl/wo_text_vit/BMLIP/results/AIBL/', 'PD_T1_'+i), os.path.join(img_path_0, i),new_spacing=new_spacing_0)

    psnr_0_1_list.append(psnr(pred_0_1, img_vol_1))
    psnr_1_0_list.append(psnr(pred_1_0, img_vol_0))
    ssim_0_1_list.append(structural_similarity(pred_0_1, img_vol_1, data_range=1))
    ssim_1_0_list.append(structural_similarity(pred_1_0, img_vol_0, data_range=1))

    print('psnr0-1', np.mean(psnr_0_1_list), '±', np.std(psnr_0_1_list))
    print('psnr1-0', np.mean(psnr_1_0_list), '±', np.std(psnr_1_0_list))
    print('ssim0-1', np.mean(ssim_0_1_list), '±', np.std(ssim_0_1_list))
    print('ssim1-0', np.mean(ssim_1_0_list), '±', np.std(ssim_1_0_list))


