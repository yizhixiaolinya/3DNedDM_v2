"""
用于对数据进行pair,找到T1和SWI共有的被试,并保存到data_try_lx/train文件夹中
"""
import os

def save_subjects_to_files(T1_img_path, T1_text_path, SWI_img_path, SWI_text_path, output_dir):
    # 读取 T1 文本文件
    with open(T1_text_path, 'r') as f:
        T1_data = f.readlines()
    
    # 读取 SWI 文本文件
    with open(SWI_text_path, 'r') as f:
        SWI_data = f.readlines()
    
    # 提取被试 ID 和对应的行
    T1_subjects = {line.split(":")[0].strip('"'): line.strip() for line in T1_data}
    SWI_subjects = {line.split(":")[0].strip('"'): line.strip() for line in SWI_data}
    
    # 找到共同的被试
    common_subjects = set(T1_subjects.keys()).intersection(SWI_subjects.keys())
    
    # 准备输出文件路径
    T1_prompt_file = os.path.join(output_dir, "train_T1_prompt.txt")
    SWI_prompt_file = os.path.join(output_dir, "train_SWI_prompt.txt")
    T1_img_file = os.path.join(output_dir, "train_T1_img.txt")
    SWI_img_file = os.path.join(output_dir, "train_SWI_img.txt")

    # 保存路径及被试名
    with open(T1_prompt_file, 'w') as t1_prompt, open(SWI_prompt_file, 'w') as swi_prompt, \
         open(T1_img_file, 'w') as t1_img, open(SWI_img_file, 'w') as swi_img:
        
        for subject in common_subjects:
            # 读取 T1 和 SWI 的行
            T1_line = T1_subjects[subject]
            SWI_line = SWI_subjects[subject]
            
            # 生成完整路径
            T1_img_full_path = os.path.join(T1_img_path, f"{subject}")
            SWI_img_full_path = os.path.join(SWI_img_path, f"{subject}")
            
            # 保存文本信息到文件
            t1_prompt.write(f"{T1_line}\n")
            swi_prompt.write(f"{SWI_line}\n")
            
            # 保存图像路径到文件
            t1_img.write(f"{T1_img_full_path}\n")
            swi_img.write(f"{SWI_img_full_path}\n")

T1_img_path = "/public_bme/data/ylwang/OASIS_clip/T1_brain/"
T1_text_path = "/public_bme/data/ylwang/OASIS_clip/OASIS_T1_vol_new_v3_full.txt"
SWI_img_path = "/public_bme/data/ylwang/OASIS_clip/SWI_brain/"
SWI_text_path = "/public_bme/data/ylwang/OASIS_clip/OASIS_SWI_vol_new_v3_full.txt"
output_dir = "/home_data/home/linxin2024/code/3DMedDM_v2/data_try_lx/train/"  # 指定输出路径

save_subjects_to_files(T1_img_path, T1_text_path, SWI_img_path, SWI_text_path, output_dir)
