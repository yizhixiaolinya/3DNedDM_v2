step 1: pair.py整理T1和SWI数据和文本,保存到data_try_lx/train文件夹中
step 2: 修改/home_data/home/linxin2024/code/3DMedDM_v2/configs/train_lccd_sr.yaml中的路径名称
step 3: train.py
step 4: 对/home_data/home/linxin2024/code/3DMedDM_v2/save/_train_lccd_sr/log.txt进行绘制
step 5: demo.py 用于test,sbatch /home_data/home/linxin2024/code/3DMedDM_v2/lx_synthesis.bash对任务进行提交
