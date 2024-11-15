# 此脚本用于检查img和txt是否一一对应
# 定义文件路径
img_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers/all_paired/train/modality2_img.txt'
prompt_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers/all_paired/train/modality2_prompt.txt'
output_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/lx/check_img_txt_result.txt'

# 读取img文件中的ID
with open(img_file_path, 'r') as img_file:
    img_ids = {line.strip().split('/')[-1] for line in img_file}

# 读取prompt文件中的ID
with open(prompt_file_path, 'r') as prompt_file:
    prompt_ids = {line.split(":")[0].strip('"') for line in prompt_file}

# 检查是否配对并保存结果
with open(output_file_path, 'w') as output_file:
    if img_ids == prompt_ids:
        output_file.write("两个文件的ID完全配对。\n")
    else:
        # 找出不匹配的ID
        missing_in_img = prompt_ids - img_ids
        missing_in_prompt = img_ids - prompt_ids

        if missing_in_img:
            output_file.write("以下ID在img文件中缺失：\n")
            for id in missing_in_img:
                output_file.write(f"{id}\n")
        if missing_in_prompt:
            output_file.write("以下ID在prompt文件中缺失：\n")
            for id in missing_in_prompt:
                output_file.write(f"{id}\n")

print(f"检查结果已保存到 {output_file_path}")
