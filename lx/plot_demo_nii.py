import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def load_nii_file(nii_path):
    """加载.nii.gz文件并返回numpy数组"""
    img = nib.load(nii_path)
    data = img.get_fdata()
    return data

def plot_slices(volume, nii_filename, slice_index=None, output_dir=None, type_suffix=""):
    """
    可视化3D MRI数据的横断面、冠状面和矢状面切片，并将图像保存为文件。
    volume: 3D MRI图像数据
    nii_filename: 文件名，用于保存图像时使用
    slice_index: (optional) 指定要显示的切片索引，默认显示中间切片
    output_dir: 保存图像的目录
    type_suffix: 文件类型后缀，用于区分T1和SWI图像
    """
    # 如果没有提供切片索引，默认使用中间切片
    if slice_index is None:
        slice_index = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成保存文件名的前缀
    filename_prefix = os.path.splitext(os.path.basename(nii_filename))[0] + type_suffix

    # 可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 显示横断面切片（axial）
    axes[0].imshow(volume[slice_index[0], :, :], cmap="gray")
    axes[0].set_title(f'Axial Slice {slice_index[0]}')
    axes[0].axis('off')
    # fig.savefig(os.path.join(output_dir, f"{filename_prefix}_axial_slice_{slice_index[0]}.png"))

    # 显示冠状面切片（coronal）
    axes[1].imshow(volume[:, slice_index[1], :], cmap="gray")
    axes[1].set_title(f'Coronal Slice {slice_index[1]}')
    axes[1].axis('off')
    # fig.savefig(os.path.join(output_dir, f"{filename_prefix}_coronal_slice_{slice_index[1]}.png"))

    # 显示矢状面切片（sagittal）
    axes[2].imshow(volume[:, :, slice_index[2]], cmap="gray")
    axes[2].set_title(f'Sagittal Slice {slice_index[2]}')
    axes[2].axis('off')
    fig.savefig(os.path.join(output_dir, f"{filename_prefix}_sagittal_slice_{slice_index[2]}.png"))

    plt.tight_layout()
    plt.close(fig)  # 关闭图像，避免内存泄露


def visualize_nii_file(nii_file_path, output_dir, type_suffix=""):
    """加载并可视化 .nii.gz 文件"""
    if not os.path.exists(nii_file_path):
        print(f"File {nii_file_path} does not exist!")
        return

    # 加载nii文件
    volume_data = load_nii_file(nii_file_path)
    print(f"Loaded volume with shape: {volume_data.shape}")

    # 可视化并保存图像
    plot_slices(volume_data, nii_file_path, output_dir=output_dir, type_suffix=type_suffix)

if __name__ == "__main__":
    # 输入要可视化的.nii.gz文件路径
    nii_pre_T1 = '/home_data/home/linxin2024/code/3DMedDM_v2/save/demo/clip_L1_loss/T12SWI_0.nii.gz'
    nii_pre_SWI = '/home_data/home/linxin2024/code/3DMedDM_v2/save/demo/clip_L1_loss/SWI2T1_0.nii.gz'
    nii_T1 = '/public_bme/data/ylwang/OASIS_clip/T1_brain/OAS30726_MR_d0061.nii.gz'
    nii_SWI = '/public_bme/data/ylwang/OASIS_clip/SWI_brain/OAS30726_MR_d0061.nii.gz'
    output_dir = '/home_data/home/linxin2024/code/3DMedDM_v2/lx/output_slices/550epoch'

    # 可视化并保存每个文件的切片图像，添加文件类型后缀
    visualize_nii_file(nii_pre_T1, output_dir, type_suffix="_pred")
    visualize_nii_file(nii_pre_SWI, output_dir, type_suffix="_pred")
    visualize_nii_file(nii_T1, output_dir, type_suffix="_T1")
    visualize_nii_file(nii_SWI, output_dir, type_suffix="_SWI")
