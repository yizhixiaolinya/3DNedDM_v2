#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=synthesis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00
#SBATCH --output=/home_data/home/linxin2024/code/3DMedDM_v2/save/bash/run.out
#SBATCH --error=/home_data/home/linxin2024/code/3DMedDM_v2/save/bash/run.err

# 创建日志目录（如果不存在）
mkdir -p /home_data/home/linxin2024/code/3DMedDM_v2/save/bash

# 激活conda环境
source /home_data/home/linxin2024/anaconda3/bin/activate synthesis

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 进入工作目录
cd /home_data/home/linxin2024/code/3DMedDM_v2/

# 检查Python环境

# 执行任务
python train.py