import torch.nn as nn
from models import register

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    # 默认的卷积操作，使用了 2D 卷积层（Conv2d）
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    # 通过跳跃连接（skip connection）使网络能够更深而不会出现梯度消失的问题
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
@register('fusionnet-resnet')
class FusionNet(nn.Module):
    # 通过堆叠多个残差块来构建。它使用卷积网络的基本原理对输入数据进行特征提取，并通过残差块保留更多细节信息
    def __init__(self, in_dim, out_dim, n_resblocks, conv=default_conv):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_resblocks = n_resblocks
        net = [
            ResBlock(
                conv, self.in_dim
            ) for _ in range(n_resblocks)
        ]
        net.append(conv(self.in_dim, self.out_dim, 3))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
