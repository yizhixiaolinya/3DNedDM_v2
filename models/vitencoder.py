# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace
from einops import rearrange
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from models import register


def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')


def to_4d(x, d, h, w):
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), d, h, w)

class DepthWiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=True):
        super(DepthWiseConv3d, self).__init__()
        self.net = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, stride=stride, bias=bias),
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.net(x)

class Mlp_our(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()
        self.alpha = nn.Parameter(torch.ones(hidden_features, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(hidden_features, 1, 1, 1))
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x * self.alpha + self.beta) * x
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_restormer(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False):
        super(Mlp_restormer, self).__init__()

        self.project_in = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv3d(hidden_features // 2, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, C, D, H, W = x.shape
        N = D * H * W
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, C, D, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., drop=0., attn_drop=0., LayerNorm_type='WithBias'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.self_attn = WindowAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = LayerNorm(self.dim, LayerNorm_type)
        self.norm2 = LayerNorm(self.dim, LayerNorm_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_our(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MAE(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """
    def __init__(self, args):
        super(MAE, self).__init__()
        self.args = args
        pretrained: str
        in_channels: int=args.n_colors
        patch_size: tuple = (4, 4, 4)
        hidden_size: int = args.hidden_size
        num_layers: int = args.num_layers
        num_heads: int = args.num_heads
        decoder_dim: int = args.decoder_dim
        decoder_depth: int = args.decoder_depth
        decoder_heads: int = args.decoder_heads
        self.out_dim = args.out_dim

        self.encoder = nn.ModuleList(
            [
                SwinTransformerBlock(
                    hidden_size, num_heads, mlp_ratio=4., drop=0., attn_drop=0., LayerNorm_type='WithBias'
                ) for i in range(num_layers)
            ])

        self.to_patch = PatchEmbed(kernel_size=patch_size,
        stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2),
        padding=(1, 1, 1),
        in_chans=in_channels,
        embed_dim=hidden_size)

        self.enc_to_dec = (
            nn.Conv3d(hidden_size, decoder_dim, kernel_size=1)
            if hidden_size != decoder_dim
            else nn.Identity()
        )

        # build up decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    decoder_dim, decoder_heads, mlp_ratio=4., drop=0., attn_drop=0., LayerNorm_type='WithBias'
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = LayerNorm(decoder_dim, LayerNorm_type='WithBias')

        self.patch_size = patch_size
        self.last_expand = 64
        self.to_pixels_v1 = nn.Linear(decoder_dim, self.last_expand)
        self.output_v1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=decoder_dim, out_channels=decoder_dim, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm3d(decoder_dim),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=decoder_dim, out_channels=self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.out_dim)
        )

        self.hidden_size = hidden_size
    def forward(self, x):
        # get patches
        B, _, D, H, W = x.shape
        patches = self.to_patch(x)   #patches.shape=(12, 64, 216)

        for blk in self.encoder:
            tokens = blk(patches)

        encoded_tokens = tokens
        #decoder_tokens = self.enc_to_dec(encoded_tokens)

        for blk in self.decoder_blocks:
            decoder_tokens = blk(encoded_tokens)
        decoded_tokens = self.decoder_norm(decoder_tokens)

        x = self.output_v1(decoded_tokens)
        return x, patches #, batch_range, masked_indices

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple = (4, 4),
        stride: tuple = (2, 2),
        padding: tuple = (1, 1),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim // 3),
            nn.Conv3d(embed_dim // 3, embed_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x



@register('vitencoder-B')
def make_vit(img_size=[20, 20, 20], patch_size=[5, 5, 5], num_layers=12, hidden_size=768, mlp_dim=3072,
              num_heads=12, decoder_dim=768, decoder_depth=1, decoder_heads=12, out_dim =256, no_upsampling=True):
    args = Namespace()
    args.img_size = img_size
    args.patch_size = patch_size
    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.mlp_dim = mlp_dim
    args.num_heads = num_heads
    args.decoder_dim = decoder_dim
    args.decoder_depth = decoder_depth
    args.decoder_heads = decoder_heads
    args.no_upsampling = no_upsampling
    args.n_colors = 1
    args.out_dim = out_dim
    return MAE(args)


@register('vitencoder-L')
def make_vit(img_size=[24, 24, 24], patch_size=[6, 6, 6], num_layers=24, hidden_size=1024, mlp_dim=4096,
              num_heads=16, decoder_dim=1024, decoder_depth=1, decoder_heads=8, out_dim = 1024, no_upsampling=True):
    args = Namespace()
    args.img_size = img_size
    args.patch_size = patch_size
    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.mlp_dim = mlp_dim
    args.num_heads = num_heads
    args.decoder_dim = decoder_dim
    args.decoder_depth = decoder_depth
    args.decoder_heads = decoder_heads
    args.no_upsampling = no_upsampling
    args.n_colors = 1
    args.out_dim = out_dim
    return MAE(args)


