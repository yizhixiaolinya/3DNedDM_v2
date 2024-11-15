import pandas as pd

# 读取文件路径
txt_file_path = 'E:/Pycharm_project/3DMedDM_v2/data_try_lx/train/train_SWI_img.txt'
csv_file_path = 'E:/Pycharm_project/3DMedDM_v2/data_try_lx/OASIS3_from_clinica.csv'

# 读取 txt 文件中的文件名
with open(txt_file_path, 'r') as file:
    txt_filenames = [line.strip().split('/')[-1] for line in file.readlines()]

# 读取 CSV 文件
csv_data = pd.read_csv(csv_file_path)

# 构造 CSV 中的文件名
csv_data['filename'] = csv_data.apply(lambda x: f"{x['sub_id'][4:]}_MR_{x['source_session_id']}.nii.gz", axis=1)

# 筛选出匹配的文件名和疾病分类
matched_data = csv_data[csv_data['filename'].isin(txt_filenames)]
disease_counts = matched_data['dx1'].value_counts()

print("疾病分类统计：")
print(disease_counts)

# 找到 txt 中的文件名，但不在 CSV 中的文件名
unmatched_filenames = [filename for filename in txt_filenames if filename not in csv_data['filename'].values]

# 输出结果
print("不在 CSV 中的文件名：")
print(unmatched_filenames)

# 可选择将结果保存到新的 txt 文件
unmatched_output_path = 'E:/Pycharm_project/3DMedDM_v2/data_try_lx/unmatched_filenames_train.txt'
with open(unmatched_output_path, 'w') as output_file:
    for filename in unmatched_filenames:
        output_file.write(filename + '\n')

print(f"不在 CSV 中的文件名已保存至 {unmatched_output_path}")