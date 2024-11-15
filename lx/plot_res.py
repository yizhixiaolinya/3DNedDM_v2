import nibabel as nib
import numpy as np

# 读取两个 .nii.gz 文件路径
file_path1 = '/public_bme/data/ylwang/OASIS_clip/T1_brain/OAS30726_MR_d0061.nii.gz'
file_path2 = '/public_bme/data/ylwang/OASIS_clip/SWI_brain/OAS30726_MR_d0061.nii.gz'

# 加载图像数据
img1 = nib.load(file_path1)
img2 = nib.load(file_path2)

# 提取图像数据数组
data1 = img1.get_fdata()
data2 = img2.get_fdata()

# 计算残差
residual = data1 - data2

# 创建残差图像对象
residual_img = nib.Nifti1Image(residual, img1.affine, img1.header)
# 保存残差图像
output_path = '/home_data/home/linxin2024/code/3DMedDM_v2/save/residual_image.nii.gz'
nib.save(residual_img, output_path)

print(f"残差图像已保存至: {output_path}")
