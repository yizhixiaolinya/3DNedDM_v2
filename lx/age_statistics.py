import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'C:/Users/Admin/Desktop/文件/ISMRM/UKBiobank/ukb669433.csv'
data = pd.read_csv(file_path, usecols=['eid', '21022-0.0'])  # 读取 eid 和年龄列

# 移除重复的 eid，确保每个 eid 只保留一个记录
unique_data = data.drop_duplicates(subset='eid')

# 统计行数（唯一被试数目）
num_unique_subjects = unique_data.shape[0]

# 计算年龄的描述性统计
age_stats = unique_data['21022-0.0'].describe()

# 将统计结果保存到指定路径
output_path = 'E:/Pycharm_project/3DMedDM_v2/lx/output.txt'
with open(output_path, 'w') as f:
    f.write("Age Statistics Distribution (Unique IDs):\n")
    f.write(age_stats.to_string())  # 将统计结果转换为字符串并写入文件
    f.write(f"\nNumber of unique subjects: {num_unique_subjects}")  # 写入行数信息

# 绘制唯一 ID 对应的年龄分布的直方图并保存
plt.figure(figsize=(10, 6))
plt.hist(unique_data['21022-0.0'].dropna(), bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Numbers of Subjects')
plt.title('Age Distribution (Unique IDs)')
plt.savefig(output_path.replace('.txt', '_age_distribution.png'))  # 保存直方图图像
plt.show()
