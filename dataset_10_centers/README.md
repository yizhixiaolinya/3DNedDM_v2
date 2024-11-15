1. /home_data/home/linxin2024/code/3DMedDM_v2/dataset 包含了10个cohorts文件夹

每个cohorts文件夹中又包含了modality1_modality2文件夹

每个modality1_modality2文件夹中包含了四个txt文件，分别为modality1_img, modality2_img, modality1_prompt, modality2_prompt
(Pay Attention: HPCD和ADNI在服务器上的路径又分为train和test，在整合中需要注意)

2. 另外，/home_data/home/linxin2024/code/3DMedDM_v2/dataset/divide_by_modality包含了按模态整合后的train和test文件夹

3. 最后，/home_data/home/linxin2024/code/3DMedDM_v2/dataset/all_paired包含了按train、val和test整合后的四个txt文件（不划分具体配对模态）

其中，val是从train中每种配对的模态随机取30个得到的。