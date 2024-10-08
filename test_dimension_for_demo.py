# SimpleITK Image Dimensions: (21, 432, 432)
# Nibabel Image Dimensions: (432, 432, 21)

import SimpleITK as sitk
import nibabel as nib
import torch
from utils_clip.simple_tokenizer import SimpleTokenizer
from CLIP.model import CLIP
from utils_clip import load_config_file
import os

# 初始化 CLIP 模型和 tokenizer
checkpoint_path = '/public_bme/home/linxin/13119176/checkpoint_CLIP.pt'
MODEL_CONFIG_PATH = '/home_data/home/linxin2024/code/3DMedDM_v2/CLIP/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

tokenizer = SimpleTokenizer()
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

# 读取图像文件路径
image_path = r'/public_bme/data/ylwang/15T_3T/img/Philip_T1.nii.gz'

# 读取文本路径
prompt_M1 = r'/public_bme/data/ylwang/15T_3T/text_prompt_HCPA.txt'
with open(prompt_M1, 'r') as f:
    lines = f.readlines()

# 假设读取第一行作为文本示例
text_prompt_M1 = lines[0].split(':', 1)[1].strip()

# 1. 使用 SimpleITK 读取图像
sitk_image = sitk.ReadImage(image_path)
sitk_image_array = sitk.GetArrayFromImage(sitk_image)

# 输出 SimpleITK 图像维度
print(f"SimpleITK Image Dimensions: {sitk_image_array.shape}")

# 2. 使用 nibabel 读取图像
nib_image = nib.load(image_path)
nib_image_array = nib_image.get_fdata()

# 输出 nibabel 图像维度
print(f"Nibabel Image Dimensions: {nib_image_array.shape}")

# 3. 处理文本并输出 token 数量
def tokenize_and_encode(text, tokenizer, model):
    # Tokenize 文本
    seq = tokenizer.encode(text)
    
    # 打印 token 的数量
    print(f"Number of tokens in text: {len(seq)}")

    # 构建 tensor 并进行编码
    seq_tensor = torch.tensor([seq])
    with torch.no_grad():
        seq_encoded = model.encode_text(seq_tensor)
    return seq_encoded

# 处理并输出文本维度
seq_src = tokenize_and_encode(text_prompt_M1, tokenizer, model)

print(f"Text Source Embedding Dimensions: {seq_src.shape}")
