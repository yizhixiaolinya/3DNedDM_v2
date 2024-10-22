import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置文件路径
log_file_path = '/home_data/home/linxin2024/code/3DMedDM_v2/save/train/_train_lccd_sr/log.txt'

# 初始化存储数据的列表
epochs = []
loss0 = []
loss1 = []

# 读取日志文件
with open(log_file_path, 'r') as file:
    for line in file:
        # 检查是否是包含loss信息的行
        if 'epoch' in line and 'loss0' in line and 'loss1' in line:
            parts = line.split(',')
            epoch_part = parts[0].split('/')[0].strip().split(' ')[1]  # 提取epoch
            loss0_part = float(parts[2].split('=')[1].strip())  # 提取loss0
            loss1_part = float(parts[3].split('=')[1].strip())  # 提取loss1
            
            epochs.append(int(epoch_part))
            loss0.append(loss0_part)
            loss1.append(loss1_part)

# 转换为numpy数组
epochs = np.array(epochs)
loss0 = np.array(loss0)
loss1 = np.array(loss1)

# 绘制Loss0曲线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss0, label='Loss0', color=(236/255, 166/255, 128/255), marker='o')
plt.title('Loss0_curve')
plt.xlabel('Epoch')
plt.ylabel('Loss0')
plt.grid()
plt.xticks(np.arange(0, max(epochs)+1, 50))
plt.tight_layout()
plt.savefig('loss0_curve.png')  # 保存图表为PNG文件
plt.show()  # 显示图表

# 绘制Loss1曲线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss1, label='Loss1', color=(122/255, 199/255, 226/255), marker='o')
plt.title('Loss1_curve')
plt.xlabel('Epoch')
plt.ylabel('Loss1')
plt.grid()
plt.xticks(np.arange(0, max(epochs)+1, 50))
plt.tight_layout()
plt.savefig('loss1_curve.png')  # 保存图表为PNG文件
plt.show()  # 显示图表
