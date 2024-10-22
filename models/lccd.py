import torch.nn as nn
from models.cross_att import Basic_block
from models.linear import Linear # 用于判别
from models.diffusion import UNet, Diffusion # U-Net和 Diffusion
from models import register
@register('lccd')
class LCCD(nn.Module):
    # 使用扩散模型（Diffusion）生成图像，并通过 U-Net 处理噪声图像以生成目标和源图像
    def __init__(self, encoder_spec, no_imnet):
        super().__init__()
        self.diffusion = Diffusion()
        self.unet = UNet()
    def forward(self, src_lr, tgt_lr, prompt_src, prompt_tgt):

        #tarin together
        t_1 = self.diffusion.sample_timesteps(src_lr.shape[0]).cuda() # 控制噪声程度
        tgt_t = self.diffusion.noise_images(tgt_lr.unsqueeze(1), src_lr.unsqueeze(1), t_1)
        tgt_out = self.unet(tgt_t, t_1, prompt_tgt)

        t_2 = self.diffusion.sample_timesteps(tgt_lr.shape[0]).cuda()
        tgt_t = self.diffusion.noise_images(src_lr.unsqueeze(1), tgt_lr.unsqueeze(1), t_2)
        src_out = self.unet(tgt_t, t_2, prompt_src)


        return tgt_out, src_out

"""
        #test forward
        
        N, K = coord_hr.shape[:2]
        feat_src_lr = self.gen_feat(src_lr)
        feat_src_lr_tgt = self.fusion(feat_src_lr, prompt_tgt)
        vector_src_tgt = F.grid_sample(feat_src_lr_tgt, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_src_tgt_with_coord = torch.cat([vector_src_tgt, coord_hr], dim=-1)
        pre_src_tgt = self.imnet(vector_src_tgt_with_coord.view(N * K, -1)).view(N, K, -1)

        
        return pre_src_tgt #, pre_src_tgt, feat_src_lr, feat_src_lr
"""



                
# Defines the PatchGAN discriminator with the specified arguments.
@register('NLDiscri')
class NLayerDiscriminator(nn.Module):
    # 实现了一个基于 MLP（多层感知器） 的判别器，用于区分生成的图像与真实图像的特征
    def __init__(self, in_dim=0, out_dim=0, hidden_list=[]):
        super().__init__()
        self.layers = Linear(in_dim=in_dim, out_dim=out_dim, hidden_list=hidden_list)
        """
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=drop))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    """
    def forward(self, src_gen, tgt_gen):
        dim = src_gen.shape[1]
        x = src_gen.reshape(-1, dim)
        y = tgt_gen.reshape(-1, dim)

        return self.layers(x).unsqueeze(-1), self.layers(y).unsqueeze(-1)
       
