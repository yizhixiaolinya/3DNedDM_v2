import os

def consolidate_all_paired_data(base_dataset_path, output_base_path):
    """
    将 dataset/train/（或 dataset/test/）下的所有配对模态数据整合到一起。
    """
    for split in ['train', 'test']:
        dataset_path = os.path.join(base_dataset_path, split)
        output_path = os.path.join(output_base_path, split)
        os.makedirs(output_path, exist_ok=True)

        modality1_img_lines = []
        modality1_prompt_lines = []
        modality2_img_lines = []
        modality2_prompt_lines = []

        # 遍历配对模态文件夹
        if not os.path.exists(dataset_path):
            print(f"{dataset_path} 不存在，跳过。")
            continue

        pair_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        for pair_folder in pair_folders:
            pair_folder_path = os.path.join(dataset_path, pair_folder)
            # 获取该文件夹中的所有文件
            files = os.listdir(pair_folder_path)
            # 找到两个模态的名称
            img_files = [f for f in files if f.endswith('_img.txt')]
            prompt_files = [f for f in files if f.endswith('_prompt.txt')]

            if len(img_files) != 2 or len(prompt_files) != 2:
                print(f"{pair_folder_path} 中的文件不完整，跳过。")
                continue

            # 假设文件命名为 <modality>_img.txt 和 <modality>_prompt.txt
            modalities = [f[:-8] for f in img_files]  # 去掉 '_img.txt' 得到模态名称

            # 读取第一个模态的数据
            mod1 = modalities[0]
            mod1_img_file = os.path.join(pair_folder_path, f"{mod1}_img.txt")
            mod1_prompt_file = os.path.join(pair_folder_path, f"{mod1}_prompt.txt")
            with open(mod1_img_file, 'r') as f:
                mod1_img_lines = [line.strip() for line in f if line.strip()]
            with open(mod1_prompt_file, 'r') as f:
                mod1_prompt_lines = [line.strip() for line in f if line.strip()]
            modality1_img_lines.extend(mod1_img_lines)
            modality1_prompt_lines.extend(mod1_prompt_lines)

            # 读取第二个模态的数据
            mod2 = modalities[1]
            mod2_img_file = os.path.join(pair_folder_path, f"{mod2}_img.txt")
            mod2_prompt_file = os.path.join(pair_folder_path, f"{mod2}_prompt.txt")
            with open(mod2_img_file, 'r') as f:
                mod2_img_lines = [line.strip() for line in f if line.strip()]
            with open(mod2_prompt_file, 'r') as f:
                mod2_prompt_lines = [line.strip() for line in f if line.strip()]
            modality2_img_lines.extend(mod2_img_lines)
            modality2_prompt_lines.extend(mod2_prompt_lines)

            print(f"已处理 {pair_folder_path}")

        # 将整合的数据写入输出文件
        modality1_img_file = os.path.join(output_path, 'modality1_img.txt')
        modality1_prompt_file = os.path.join(output_path, 'modality1_prompt.txt')
        modality2_img_file = os.path.join(output_path, 'modality2_img.txt')
        modality2_prompt_file = os.path.join(output_path, 'modality2_prompt.txt')

        with open(modality1_img_file, 'w') as f:
            f.write('\n'.join(modality1_img_lines))
        with open(modality1_prompt_file, 'w') as f:
            f.write('\n'.join(modality1_prompt_lines))
        with open(modality2_img_file, 'w') as f:
            f.write('\n'.join(modality2_img_lines))
        with open(modality2_prompt_file, 'w') as f:
            f.write('\n'.join(modality2_prompt_lines))

        print(f"{split} 数据已整合到 {output_path}")

def main():
    # 基础数据集路径
    base_dataset_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset/divide_by_modality'
    # 输出路径
    output_base_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset/all_paired'

    # 整合数据
    consolidate_all_paired_data(base_dataset_path, output_base_path)

if __name__ == "__main__":
    main()
