import os
import itertools
import re


def read_cohorts_file(cohorts_file_path):
    """
    读取 Cohorts.txt 文件，解析每个 cohort 的信息。
    """
    cohorts = []
    with open(cohorts_file_path, 'r') as f:
        content = f.read()
        # 首先获取 cohort_name 列表
        cohort_names = re.findall(r"cohort_name\s*=\s*\[(.*?)\];", content, re.DOTALL)
        if cohort_names:
            cohort_list = re.findall(r"'(.*?)'", cohort_names[0])
            for cohort_name in cohort_list:
                # 针对每个 cohort，提取其配置信息
                pattern = rf"{cohort_name}:\s*(.*?)\n\s*dataset_type\s*=\s*'?(.*?)'?\s*;"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    config_str = match.group(1)
                    dataset_type = match.group(2) if match.group(2) else 'default'
                    path = re.findall(r"path\s*=\s*'(.*?)'", config_str)
                    folders = re.findall(r"folders\s*=\s*\[(.*?)\]", config_str)
                    txt_files = re.findall(r"txt_files\s*=\s*\[(.*?)\]", config_str)
                    path_train = re.findall(r"path_train\s*=\s*'(.*?)'", config_str)
                    txt_files_train = re.findall(r"txt_files_train\s*=\s*\[(.*?)\]", config_str)
                    path_test = re.findall(r"path_test\s*=\s*'(.*?)'", config_str)
                    txt_files_test = re.findall(r"txt_files_test\s*=\s*\[(.*?)\]", config_str)

                    cohort_info = {
                        'name': cohort_name,
                        'dataset_type': dataset_type,
                    }
                    if path:
                        cohort_info['path'] = path[0]
                    if folders:
                        # 解析 folders 列表
                        cohort_info['folders'] = [s.strip().strip("'\"") for s in folders[0].split(',')]
                    if txt_files:
                        cohort_info['txt_files'] = [s.strip().strip("'\"") for s in txt_files[0].split(',')]
                    if path_train:
                        cohort_info['path_train'] = path_train[0]
                    if txt_files_train:
                        cohort_info['txt_files_train'] = [s.strip().strip("'\"") for s in txt_files_train[0].split(',')]
                    if path_test:
                        cohort_info['path_test'] = path_test[0]
                    if txt_files_test:
                        cohort_info['txt_files_test'] = [s.strip().strip("'\"") for s in txt_files_test[0].split(',')]
                    cohorts.append(cohort_info)
        else:
            print("未找到 cohort_name 列表。")
    return cohorts


def standardize_filename(filename, dataset_type):
    """
    标准化文件名，提取配对标识符。
    不同数据集使用不同的标准化规则。
    """
    if dataset_type in ['BCP', 'HCPA', 'HCPD']:
        # 提取形如 MNBCP000080-43mo 或 HCA6002236 的前缀作为配对标识符
        match = re.match(r"(MNBCP\d+-\d+mo|HCA\d+|HCD\d+)", filename)
        if match:
            return match.group(1)
    elif dataset_type == 'UKBiobank':
        # 对于 UK Biobank, 提取主要的 ID 并忽略 20252/20253 标识
        match = re.match(r"(\d+)_2025[2|3]", filename)
        if match:
            return match.group(1)
    elif dataset_type == 'OASIS3':
        # 对于 OASIS3，提取 OAS 开头的 ID
        match = re.match(r"(OAS\d+)_.*", filename)
        if match:
            return match.group(1)
    else:
        # 默认情况下返回原始文件名
        return filename
    return filename  # 如果不匹配，返回原始文件名


def extract_data_filenames_from_txt(txt_file_path):
    """
    从 txt 文件中提取数据文件名和对应的内容。
    假设 txt 文件的每一行格式为：
    "文件名": "其他信息"
    """
    data = {}
    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    filename_with_quotes = parts[0].strip()
                    content = parts[1].strip()
                    filename = filename_with_quotes.strip('"')
                    data[filename] = content.strip('"')
    return data

def map_folder_to_modality(folder_name):
    """
    将文件夹名称映射到标准模态名称。
    """
    mapping = {
        'flair_brain': 'FLAIR',
        'FLAIR': 'FLAIR',
        'SWI_brain': 'SWI',
        'SWI': 'SWI',
        'T1_brain': 'T1w',
        'T1w': 'T1w',
        'T1_pad': 'T1w',
        'T1': 'T1w',
        'T2': 'T2w',
        'T2_brain': 'T2w',
        'T2w': 'T2w',
        'T2_pad': 'T2w',
        'FLAIR_brain': 'FLAIR',
    }
    return mapping.get(folder_name, folder_name)


def process_cohort(cohort, base_output_path):
    """
    处理单个 cohort，生成所需的文件夹和文件。
    """
    # 判断是单路径还是 train/test 路径
    if 'path' in cohort:
        paths = {'': {'path': cohort['path'], 'txt_files': cohort['txt_files']}}
    else:
        paths = {}
        if 'path_train' in cohort:
            paths['train'] = {'path': cohort['path_train'], 'txt_files': cohort['txt_files_train']}
        if 'path_test' in cohort:
            paths['test'] = {'path': cohort['path_test'], 'txt_files': cohort['txt_files_test']}
    for split_name, split_info in paths.items():
        base_path = split_info['path']
        txt_files = split_info['txt_files']
        folders = cohort['folders']
        dataset_type = cohort['dataset_type']

        # 针对 HCPD 数据集的特殊处理
        if cohort['name'] == 'HCPD' and split_name == 'train':
            txt_base_path = '/public_bme/data/ylwang/HCPD_clip/'  # txt 文件所在的正确路径
        else:
            txt_base_path = base_path

        # 创建文件夹名称到标准模态名称的映射
        folder_to_modality = {folder: map_folder_to_modality(folder) for folder in folders}

        # 加载所有 txt 文件的数据
        modality_data = {}
        modality_filenames = {}
        for folder, txt_file in zip(folders, txt_files):
            txt_file_path = os.path.join(txt_base_path, txt_file)
            if os.path.exists(txt_file_path):
                data = extract_data_filenames_from_txt(txt_file_path)
                modality_data[folder] = data
                # 获取 txt 文件中的文件名集合
                txt_filenames = set(data.keys())
            else:
                # print(f"txt 文件 {txt_file_path} 不存在。")
                modality_data[folder] = {}
                txt_filenames = set()

            # 获取文件夹中的文件名集合
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                folder_filenames = set(os.listdir(folder_path))
            else:
                print(f"文件夹 {folder_path} 不存在。")
                folder_filenames = set()

            # 标准化文件名
            standardized_txt_filenames = {standardize_filename(f, dataset_type) for f in txt_filenames}
            standardized_folder_filenames = {standardize_filename(f, dataset_type) for f in folder_filenames}

            # 取文件夹和 txt 文件中都存在的文件名
            valid_filenames = standardized_txt_filenames.intersection(standardized_folder_filenames)
            modality_filenames[folder] = valid_filenames

        # 生成所有模态的两两组合
        modality_pairs = list(itertools.combinations(folders, 2))

        # 确定输出路径
        if split_name:
            cohort_output_path = os.path.join(base_output_path, cohort['name'], split_name)
        else:
            cohort_output_path = os.path.join(base_output_path, cohort['name'])
        os.makedirs(cohort_output_path, exist_ok=True)

        for folder1, folder2 in modality_pairs:
            # 获取对应的标准模态名称
            mod1_name = folder_to_modality[folder1]
            mod2_name = folder_to_modality[folder2]
            # 对模态名称进行排序，确保文件夹命名一致
            sorted_mods = sorted([mod1_name, mod2_name])
            pair_folder_name = f"{sorted_mods[0]}_{sorted_mods[1]}"
            pair_folder_path = os.path.join(cohort_output_path, pair_folder_name)
            os.makedirs(pair_folder_path, exist_ok=True)
            # 找到两个模态中共同的文件名（标准化后）
            common_ids = modality_filenames[folder1].intersection(modality_filenames[folder2])
            # 准备写入文件的数据
            mod1_img_lines = []
            mod2_img_lines = []
            mod1_prompt_lines = []
            mod2_prompt_lines = []
            for common_id in common_ids:
                # 获取原始文件名
                filename1 = None
                for fname in modality_data[folder1]:
                    if standardize_filename(fname, dataset_type) == common_id:
                        filename1 = fname
                        break
                filename2 = None
                for fname in modality_data[folder2]:
                    if standardize_filename(fname, dataset_type) == common_id:
                        filename2 = fname
                        break
                if filename1 is None or filename2 is None:
                    continue
                # 构建图像路径
                img_path1 = os.path.join(base_path, folder1, filename1)
                img_path2 = os.path.join(base_path, folder2, filename2)
                # 检查文件是否存在
                if not os.path.exists(img_path1):
                    print(f"文件不存在：{img_path1}")
                    continue
                if not os.path.exists(img_path2):
                    print(f"文件不存在：{img_path2}")
                    continue
                mod1_img_lines.append(img_path1)
                mod2_img_lines.append(img_path2)
                # 获取提示信息
                prompt1 = f'"{filename1}": "{modality_data[folder1][filename1]}"'
                prompt2 = f'"{filename2}": "{modality_data[folder2][filename2]}"'
                mod1_prompt_lines.append(prompt1)
                mod2_prompt_lines.append(prompt2)
            # 写入文件
            mod1_img_file = os.path.join(pair_folder_path, f"{sorted_mods[0]}_img.txt")
            mod2_img_file = os.path.join(pair_folder_path, f"{sorted_mods[1]}_img.txt")
            mod1_prompt_file = os.path.join(pair_folder_path, f"{sorted_mods[0]}_prompt.txt")
            mod2_prompt_file = os.path.join(pair_folder_path, f"{sorted_mods[1]}_prompt.txt")
            with open(mod1_img_file, 'w') as f:
                f.write('\n'.join(mod1_img_lines))
            with open(mod2_img_file, 'w') as f:
                f.write('\n'.join(mod2_img_lines))
            with open(mod1_prompt_file, 'w') as f:
                f.write('\n'.join(mod1_prompt_lines))
            with open(mod2_prompt_file, 'w') as f:
                f.write('\n'.join(mod2_prompt_lines))
            # 输出配对数量到 counts.txt
            counts_file = os.path.join(pair_folder_path, 'counts.txt')
            with open(counts_file, 'w') as f:
                count = len(mod1_img_lines)
                f.write(f"配对模态 {sorted_mods[0]} 和 {sorted_mods[1]} 的样本数量：{count}\n")
            print(f"已生成 {pair_folder_path} 中的文件，包含 {len(mod1_img_lines)} 个样本。")

def main():
    # 指定 Cohorts.txt 的路径
    cohorts_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers/Cohorts.txt'
    # 输出的基准路径
    base_output_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers'
    cohorts = read_cohorts_file(cohorts_file_path)
    for cohort in cohorts:
        process_cohort(cohort, base_output_path)

if __name__ == "__main__":
    main()
