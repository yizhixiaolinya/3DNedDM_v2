import torch

checkpoint_path = '/home_data/home/linxin2024/code/3DMedDM_v2/save/Loss/_train_lccd_sr_batch_size_2/epoch-last.pth'
try:
    checkpoint = torch.load(checkpoint_path)
    print("Checkpoint loaded successfully.")
except Exception as e:
    print("Failed to load checkpoint:", e)
