import os
import itertools
import re
import shutil


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
        else:
            return filename  # 如果不匹配，返回原始文件名
    elif dataset_type == 'UKBiobank':
        # 对于 UKBiobank, 提取主要的 ID 并忽略 20252/20253 标识
        match = re.match(r"(\d+)_2025[2|3]", filename)
        if match:
            return match.group(1)
        else:
            return filename
    else:
        # 默认情况下返回原始文件名
        return filename


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


def consolidate_txt_files(source_dir, target_dir, cohort_name):
    """
    将源目录中的四个 txt 文件内容整合到目标目录中对应的 txt 文件。
    """
    txt_files = ['_img.txt', '_prompt.txt']
    for file_suffix in txt_files:
        for modality in ['T1w', 'T2w', 'FLAIR', 'SWI']:
            source_file = os.path.join(source_dir, f"{modality}{file_suffix}")
            target_file = os.path.join(target_dir, f"{modality}{file_suffix}")
            if os.path.exists(source_file):
                with open(source_file, 'r') as sf, open(target_file, 'a') as tf:
                    tf.write(sf.read() + '\n')
            else:
                continue  # 如果源文件不存在，跳过


def consolidate_counts(source_dir, target_dir, cohort_name):
    """
    将源目录中的 counts.txt 内容整合到目标目录中的 counts.txt，并添加 cohort 名称。
    """
    source_counts_file = os.path.join(source_dir, 'counts.txt')
    target_counts_file = os.path.join(target_dir, 'counts.txt')
    if os.path.exists(source_counts_file):
        with open(source_counts_file, 'r') as scf, open(target_counts_file, 'a') as tcf:
            for line in scf:
                if line.strip():
                    new_line = f"{cohort_name}: {line.strip()}\n"
                    tcf.write(new_line)
    else:
        pass  # 如果源文件不存在，跳过


def process_cohorts(consolidate_order, dataset_path):
    """
    按照指定的顺序处理 Cohorts，并将配对模态的 txt 文件整合到 train 和 test 目录中。
    """
    for cohort in consolidate_order:
        cohort_name = cohort['name']
        print(f"正在处理 cohort: {cohort_name}")

        if cohort_name in ['UKBiobank', 'ADNI2']:
            # 所有数据都整合到 dataset/test/ 下
            splits = []
            if 'path' in cohort:
                splits.append('')
            if 'path_train' in cohort and 'path_test' in cohort:
                splits.extend(['train', 'test'])
            elif 'path_train' in cohort:
                splits.append('train')
            elif 'path_test' in cohort:
                splits.append('test')

            for split in splits:
                if split:
                    source_base = os.path.join(dataset_path, cohort_name, split)
                else:
                    source_base = os.path.join(dataset_path, cohort_name)
                target_base = os.path.join(dataset_path, 'test')

                if not os.path.exists(source_base):
                    print(f"源路径 {source_base} 不存在，跳过。")
                    continue

                # 遍历配对模态文件夹
                for pair_folder in os.listdir(source_base):
                    source_pair_path = os.path.join(source_base, pair_folder)
                    if os.path.isdir(source_pair_path):
                        target_pair_path = os.path.join(target_base, pair_folder)
                        os.makedirs(target_pair_path, exist_ok=True)
                        # 整合 txt 文件
                        consolidate_txt_files(source_pair_path, target_pair_path, cohort_name)
                        consolidate_counts(source_pair_path, target_pair_path, cohort_name)
                        print(f"已整合 {source_pair_path} 到 {target_pair_path}")
        else:
            # 所有数据都整合到 dataset/train/ 下
            splits = []
            if 'path' in cohort:
                splits.append('')
            if 'path_train' in cohort and 'path_test' in cohort:
                splits.extend(['train', 'test'])
            elif 'path_train' in cohort:
                splits.append('train')
            elif 'path_test' in cohort:
                splits.append('test')

            for split in splits:
                if split:
                    source_base = os.path.join(dataset_path, cohort_name, split)
                else:
                    source_base = os.path.join(dataset_path, cohort_name)
                target_base = os.path.join(dataset_path, 'train')

                if not os.path.exists(source_base):
                    print(f"源路径 {source_base} 不存在，跳过。")
                    continue

                # 遍历配对模态文件夹
                for pair_folder in os.listdir(source_base):
                    source_pair_path = os.path.join(source_base, pair_folder)
                    if os.path.isdir(source_pair_path):
                        target_pair_path = os.path.join(target_base, pair_folder)
                        os.makedirs(target_pair_path, exist_ok=True)
                        # 整合 txt 文件
                        consolidate_txt_files(source_pair_path, target_pair_path, cohort_name)
                        consolidate_counts(source_pair_path, target_pair_path, cohort_name)
                        print(f"已整合 {source_pair_path} 到 {target_pair_path}")


def main():
    # 指定 Cohorts.txt 的路径
    cohorts_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers/Cohorts.txt'
    # 指定 dataset 的基准路径
    dataset_path = '/home_data/home/linxin2024/code/3DMedDM_v2/dataset_10_centers'

    # 读取 cohorts 信息
    cohorts = read_cohorts_file(cohorts_file_path)

    # 确定拼接顺序
    cohort_order = [cohort for cohort in cohorts if cohort['name'] not in ['UKBiobank', 'ADNI2']]
    # 将 UKBiobank 和 ADNI2 放在最后
    special_cohorts = [cohort for cohort in cohorts if cohort['name'] in ['UKBiobank', 'ADNI2']]
    # 最终拼接顺序
    final_order = cohort_order + special_cohorts

    print("拼接顺序如下：")
    for idx, cohort in enumerate(final_order, 1):
        print(f"{idx}. {cohort['name']}")

    # 创建目标目录
    os.makedirs(os.path.join(dataset_path,'divide_by_modality/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path,'divide_by_modality/test'), exist_ok=True)

    # 处理并整合 cohorts
    process_cohorts(final_order, dataset_path)

    print("数据集整合完成。")


if __name__ == "__main__":
    main()
