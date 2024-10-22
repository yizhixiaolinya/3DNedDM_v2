import models
import torch
import SimpleITK as sitk
import utils
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from CLIP.model import CLIP
from utils_clip import load_config_file
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the CLIP model and tokenizer
checkpoint_path = '/public/home/v-wangyl/wo_text_vit/BMLIP/checkpoint_99_13900.pt'
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
    resample = sitk.ResampleImageFilter()
    img = sitk.GetImageFromArray(vol)
    resample.SetOutputOrigin(img_ref.GetOrigin())
    resample.SetOutputDirection(img_ref.GetDirection())
    if new_spacing is None:
        resample.SetOutputSpacing(img_ref.GetSpacing())
    else:
        resample.SetOutputSpacing(tuple(new_spacing))
    newimage = resample.Execute(img)
    sitk.WriteImage(newimage, out_path)

def set_new_spacing(ori_spacing, coord_size):
    # Calculate new spacing based on the target coordinate size and original spacing
    scale = [ori_spacing[i] / coord_size[i] for i in range(3)]
    new_spacing = tuple(scale)
    return new_spacing

def tokenize(texts, tokenizer, context_length=90):
    if isinstance(texts, str):
        texts = [texts]
    
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        token_length = len(tokens)
        # 调试输出 token 的实际长度
        # print(f"Debug: Token length for input {texts[i]} -> {token_length}")     
        if token_length > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    
    return result


def get_predictions(model, img_vol_src, img_vol_tgt, seq_src, seq_tgt):
    # 将 img_vol_src 和 img_vol_tgt 转换为 5D 张量 (batch_size, channels, depth, height, width)
    """
    train中的维度
    seq_src shape: torch.Size([2, 1, 768])
    seq_tgt shape: torch.Size([2, 1, 768])
    tgt_hr shape: torch.Size([2, 32, 32, 32])
    src_hr shape: torch.Size([2, 32, 32, 32])
    """
    print('img_vol_src shape:',img_vol_0.shape)
    print('img_vol_tgt shape:',img_vol_1.shape)
    print('seq_src shape:',seq_src.shape)
    print('seq_tgt shape:',seq_tgt.shape)
    print('done1')

    img_vol_src = torch.tensor(img_vol_src).cuda().float().unsqueeze(0).unsqueeze(0)  # (1, 1, depth, height, width)
    img_vol_tgt = torch.tensor(img_vol_tgt).cuda().float().unsqueeze(0).unsqueeze(0)  # (1, 1, depth, height, width)
    print('img_vol_src shape:',img_vol_0.shape)
    print('img_vol_tgt shape:',img_vol_1.shape)
    print('done2')

    # 调用模型进行推理
    model.eval()
    with torch.no_grad():
        # 传入的 seq_src 和 seq_tgt 应该已经是适当维度的张量
        pred_src_tgt = model(img_vol_src, img_vol_tgt, seq_src.cuda().float(), seq_tgt.cuda().float())
    
    # 移除 batch 和通道维度，将输出转化为 numpy 格式
    pred_src_tgt = pred_src_tgt.squeeze(0).squeeze(-1).cpu().numpy() 
    print('pred_src_tgt shape:',pred_src_tgt.shape)
    print('done3')
    
    return pred_src_tgt

def psnr(ref, ret):
    err = ref - ret
    return -10 * np.log10(np.mean(err ** 2))

# File paths and model loading
model_pth = '/home_data/home/linxin2024/code/3DMedDM_v2/save/_train_lccd_sr/epoch-last.pth'
model_img = models.make(torch.load(model_pth)['model_G'], load_sd=True).cuda()
# # 输出模型的参数名称和形状
# print("Model Parameters and Shapes:")
# for name, param in model_img.named_parameters():
#     if param.requires_grad:
#         print(f"Parameter name: {name}, Shape: {param.shape}")



img_path_0 = r'/public_bme/data/ylwang/15T_3T/img'
img_path_1 = r'/public_bme/data/ylwang/15T_3T/img'
img_list_0 = sorted(os.listdir(img_path_0))
img_list_1 = sorted(os.listdir(img_path_1))

# 文本格式: Line 1: "S_OAS30959_MR_d3385.nii.gz":"Age: 82; Gender: F; Scanner: 3.0 Siemens; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree): (2500.0, 2.2, 1000.0, 8.0)"
# "OAS31009_MR_d2325.nii.gz": "Age: 64; Gender: M; Scanner: 3.0 Siemens; Modality: SWI; Voxel size: (0.9, 0.9, 2.0); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree):  (28.0, 20.0, None, 15.0)"
prompt_M1 = r'/public_bme/data/ylwang/15T_3T/text_prompt_HCPA.txt'
prompt_M2 = r'/public_bme/data/ylwang/15T_3T/text_prompt_HCPA.txt'

with open(prompt_M1) as f1, open(prompt_M2) as f2:
    lines_M1 = f1.readlines()
    lines_M2 = f2.readlines()

# 存储两个图像对之间的评估结果: PSNR（峰值信噪比）和 SSIM（结构相似度）
psnr_0_1_list = []
ssim_0_1_list = []

for idx, (i, j) in enumerate(zip(img_list_0, img_list_1)):
    start_time = time.time()
    
    # Load the image volumes and their spacings
    img_0 = sitk.ReadImage(os.path.join(img_path_0, i))
    img_vol_0 = sitk.GetArrayFromImage(img_0)
    img_0_spacing = img_0.GetSpacing()

    img_1 = sitk.ReadImage(os.path.join(img_path_1, j))
    img_vol_1 = sitk.GetArrayFromImage(img_1)
    img_1_spacing = img_1.GetSpacing()

    # Clip the volumes (if necessary)
    img_vol_0 = utils.percentile_clip(img_vol_0)
    img_vol_1 = utils.percentile_clip(img_vol_1)

    coord_size = [60, 60, 60]
    coord_hr = utils.make_coord(coord_size, flatten=True)
    coord_hr = torch.tensor(coord_hr).cuda().float().unsqueeze(0)

    # Process and tokenize the text prompts
    text_src_full = lines_M1[idx].strip().replace('"', '')  # 读取完整的文本行 108 tokens
    text_tgt_full = lines_M2[idx].strip().replace('"', '')
    # 提取文件名（用于图像）和实际提示信息（用于文本处理）
    filename_src, text_src = text_src_full.split(':', 1)  # 使用 ':' 作为分隔符，将文件名与文本内容分开
    filename_tgt, text_tgt = text_tgt_full.split(':', 1)
    # 打印调试输出 82 tokens
    # print(f"Debug: filename_src -> {filename_src}, text_src -> {text_src}")

    
    seq_src = tokenize(text_src, tokenizer).cuda()
    with torch.no_grad():
        seq_src = model.encode_text(seq_src)
    
    seq_tgt = tokenize(text_tgt, tokenizer).cuda()
    with torch.no_grad():
        seq_tgt = model.encode_text(seq_tgt)
    
    # Get predictions using the model
    pred_0_1 = get_predictions(model_img, img_vol_0, img_vol_1, seq_src, seq_tgt)

    # Adjust the spacing based on the target coordinates
    new_spacing_1 = set_new_spacing(img_1_spacing, coord_size)

    # Save the predicted image
    write_img(pred_0_1, os.path.join('/public/home/v-wangyl/wo_text_vit/BMLIP/results/15T_3T/HCPA', '15T_3T_'+i),
              os.path.join(img_path_1, j), new_spacing=new_spacing_1)

    # Compute and store metrics (PSNR and SSIM)
    psnr_0_1_list.append(psnr(pred_0_1, img_vol_1))
    ssim_0_1_list.append(structural_similarity(pred_0_1, img_vol_1, data_range=1))

    # Print metrics after each iteration
    print(f"PSNR: {np.mean(psnr_0_1_list):.2f} ± {np.std(psnr_0_1_list):.2f}")
    print(f"SSIM: {np.mean(ssim_0_1_list):.4f} ± {np.std(ssim_0_1_list):.4f}")
    
    print(f"Iteration {idx} completed in {time.time() - start_time:.2f} seconds.")
