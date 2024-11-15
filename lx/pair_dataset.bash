#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=pair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH --output=/home_data/home/linxin2024/code/3DMedDM_v2/lx/pair_datasets/UKBiobank.out


# 激活 conda 环境
source /home_data/home/linxin2024/anaconda3/bin/activate synthesis

# 进入工作目录
cd /home_data/home/linxin2024/code/3DMedDM_v2/lx

# 执行任务
python /home_data/home/linxin2024/code/3DMedDM_v2/lx/pair_datasets_for_abstract.py
