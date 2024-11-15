import os
from itertools import combinations
import re  # 导入正则表达式模块


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
    else:
        # 默认情况下返回原始文件名
        return filename

def count_files_in_folders(base_path, folders):
    """
    统计指定路径下的多个文件夹中的文件数量。
    """
    file_counts = {}
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            file_counts[folder] = len(files)
        else:
            file_counts[folder] = 0
    return file_counts

def count_paired_files_with_txt(base_path, folders, txt_files, dataset_type):
    """
    统计各文件夹中的文件与txt文件中列出的文件名配对的数量，
    并返回配对的文件集合字典。
    """
    txt_data_filenames = set()
    for txt_file in txt_files:
        # txt_file_path = os.path.join('/public_bme/data/ylwang/HCPD_clip/', txt_file)
        txt_file_path = os.path.join(base_path, txt_file)
        if os.path.exists(txt_file_path):
            filenames = extract_data_filenames_from_txt(txt_file_path)
            # 将txt文件名也标准化
            txt_data_filenames.update(standardize_filename(fname, dataset_type) for fname in filenames)
        else:
            print(f"txt文件 {txt_file} 不存在。")

    paired_counts = {}
    folder_paired_files = {}  # 保存与txt文件匹配的文件名集合
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            data_files = set(os.listdir(folder_path))
            # 对文件夹内的文件名进行标准化
            standardized_files = {standardize_filename(f, dataset_type) for f in data_files}
            # 计算配对的文件名
            paired_files = standardized_files.intersection(txt_data_filenames)
            paired_counts[folder] = len(paired_files)
            folder_paired_files[folder] = paired_files  # 保存与txt匹配的文件名集合
        else:
            paired_counts[folder] = 0
            folder_paired_files[folder] = set()
    return paired_counts, folder_paired_files

def extract_data_filenames_from_txt(txt_file_path):
    """
    从txt文件中提取数据文件名。
    假设txt文件的每一行格式为：
    "文件名": "其他信息"
    """
    data_filenames = set()
    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    filename_with_quotes = parts[0].strip()
                    filename = filename_with_quotes.strip('"')
                    data_filenames.add(filename)
    return data_filenames

def count_paired_files_between_folders(folder_paired_files, folder_pairs):
    """
    统计各文件夹对之间配对的文件数量，只考虑与txt文件匹配的文件名。
    """
    pair_counts = {}
    for folder1, folder2 in folder_pairs:
        files1 = folder_paired_files.get(folder1, set())
        files2 = folder_paired_files.get(folder2, set())
        # 计算共同的文件名
        common_files = files1.intersection(files2)
        pair_counts[(folder1, folder2)] = len(common_files)
    return pair_counts

def count_files_in_all_modalities(folder_paired_files, modalities):
    """
    统计同时存在于指定模态（文件夹）中的文件数量，只考虑与txt文件匹配的文件名。
    """
    # 获取指定模态中的文件名集合
    files_sets = [folder_paired_files.get(modality, set()) for modality in modalities]
    # 计算这些集合的交集
    common_files = set.intersection(*files_sets)
    # 返回同时存在于所有指定模态中的文件数量
    return len(common_files)

def run_statistics(base_path, folders, txt_files, dataset_type):
    """
    运行统计信息并输出结果，包括：
    1. 每个文件夹中的文件数及其总和
    2. 每个文件夹中与txt文件配对的文件数及其总和
    3. 文件夹对之间配对的文件数及其总和
    4. 同时存在于所有指定模态中的文件数量
    """
    # 统计每个文件夹中的文件数
    file_counts = count_files_in_folders(base_path, folders)
    total_file_count = sum(file_counts.values())
    print("\n1. 每个文件夹中的文件数及其总和：")
    for folder, count in file_counts.items():
        print(f"{folder} 中的文件数: {count}")
    print(f"所有文件夹中的文件总数: {total_file_count}")

    # 统计每个文件夹中与txt文件配对的文件数
    paired_counts, folder_paired_files = count_paired_files_with_txt(base_path, folders, txt_files, dataset_type)
    total_paired_count = sum(paired_counts.values())
    print("\n2. 每个文件夹中与txt文件配对的文件数及其总和：")
    for folder, count in paired_counts.items():
        print(f"{folder} 中与txt文件中列出的文件名配对的文件数: {count}")
    print(f"所有文件夹与txt文件中配对的文件总数: {total_paired_count}")

    # 动态生成文件夹对
    folder_pairs = list(combinations(folders, 2))

    # 统计文件夹对之间配对的文件数
    pair_counts = count_paired_files_between_folders(folder_paired_files, folder_pairs)
    total_pair_count = sum(pair_counts.values())
    print("\n3. 文件夹对之间配对的文件数及其总和：")
    for (folder1, folder2), count in pair_counts.items():
        print(f"{folder1} 和 {folder2} 之间配对的文件数: {count}")
    print(f"所有文件夹对之间配对的文件总数: {total_pair_count}")

    # 统计同时存在于所有指定模态中的文件数量
    common_file_count = count_files_in_all_modalities(folder_paired_files, folders)
    print("\n4. 同时存在于所有指定模态中的文件数量：")
    print(f"在 {', '.join(folders)} 中同时存在的文件数: {common_file_count}")


# OASIS
# base_path = '/public_bme/data/ylwang/OASIS_clip/'
# folders = ['flair_brain', 'SWI_brain', 'T1_brain', 'T2_brain']
# txt_files = ['OASIS_SWI_vol_new_v3_full.txt', 'OASIS_FLAIR_vol_new_v3_full.txt',
#              'OASIS_T1_vol_new_v3_full.txt', 'OASIS_T2_vol_new_v3_full.txt']

# AIBL
# base_path = '/public_bme/data/ylwang/AIBL_pair/'
# folders = ['FLAIR', 'T1w', 'T2w']
# txt_files = ['FLAIR_prompt.txt', 'T1_prompt.txt', 'T2_prompt.txt']

# CBMFM
# base_path = '/public_bme/data/ylwang/CBMFM_clip/'
# folders = ['FLAIR_brain', 'T1_brain', 'T2_brain']
# txt_files = ['FLAIR_prompt_full.txt', 'T1_prompt_full.txt', 'T2_prompt_full.txt']

# IXI
# base_path = '/public_bme/data/ylwang/IXI_clip/'
# folders = ['T1_brain', 'T2_brain']
# txt_files = ['IXI_T1_vol_new_v3_full.txt', 'IXI_T2_vol_new_v3_full.txt']
# dataset_type = 'default'  # 使用默认类型

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 有一些数据集的命名不太一样：BCP HCPA HCPD UKBiobank

# # BCP
# base_path = '/public_bme/data/ylwang/BCP_clip/'
# folders = ['T1', 'T2']
# txt_files = ['BCP_T1_vol_new_v3_full.txt', 'BCP_T2_vol_new_v3_full.txt']
# dataset_type = 'BCP'  # 添加 dataset_type 参数

# # HCPA
# base_path = '/public_bme/data/ylwang/HCPA_clip/'
# folders = ['T1', 'T2']
# txt_files = ['HCPA_T1_vol_new_v3_full.txt', 'HCPA_T2_vol_new_v3_full.txt']
# dataset_type = 'HCPA'  # 添加 dataset_type 参数

# # HCPD
# dataset_type = 'HCPD'
# base_path = '/public_bme/data/ylwang/HCPD_clip/train'
# folders = ['T1w', 'T2w']
# txt_files = ['HCPD_T1_vol_new_v3_full.txt', 'HCPD_T2_vol_new_v3_full.txt']
# print('train dataset:')
# run_statistics(base_path, folders, txt_files, dataset_type)
# base_path2 = '/public_bme/data/ylwang/HCPD_clip/test'
# folders2 = ['T1w', 'T2w']
# txt_files2 = ['HCD_test_M1.txt', 'HCD_test_M2.txt']
# print('test dataset:')
# run_statistics(base_path2, folders2, txt_files2, dataset_type)

# HCPY
# base_path = '/public_bme/data/ylwang/HCPY_clip/'
# folders = ['T1_pad', 'T2_pad']
# txt_files = ['HCP_T1_vol_new_v3_full.txt', 'HCP_T2_vol_new_v3_full.txt']

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 有一些数据集的命名不太一样：BCP HCPA HCPD UKBiobank

# UK Biobank
base_path = '/public_bme/data/ylwang/UKBiobank/'
folders = ['T1', 'FLAIR']
txt_files = ['T1.txt', 'FLAIR.txt']
dataset_type = 'UKBiobank'

# ADNI-2
# base_path = '/public_bme/data/ylwang/ADNI/train/'
# folders = ['T1', 'FLAIR']
# txt_files = ['T1.txt', 'FLAIR.txt']
# print('train dataset:')
# run_statistics(base_path, folders, txt_files)
# base_path2 = '/public_bme/data/ylwang/ADNI/test/'
# folders2 = ['T1', 'FLAIR']
# txt_files2 = ['T1.txt', 'FLAIR.txt']
# print('test dataset:')
# run_statistics(base_path2, folders2, txt_files2)




# 调用运行统计信息的主函数
run_statistics(base_path, folders, txt_files, dataset_type)


