import torch
import torch.nn as nn
# from torchvision.models import resnet34
# from pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from .pvt_v2 import PyramidVisionTransformerV2
# from torchvision.models import resnet50
# from .DeiT import deit_small_patch16_224 as deit
# from .DeiT import deit_base_patch16_224 as deit_base
# from .DeiT import deit_base_patch16_384 as deit_base_384
# from .CMT import cmt_ti
from functools import partial
# from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
# import numpy as np
# import math
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
from einops import rearrange
# from mmcv.ops import DeformConv2dPack as DCN
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
    # torch.max(x, 1)计算指定维度上的最大值及其索引，返回一个元组，包含两个张量，第一个是指定维度上的最大值，第二个是最大值的索引。
    # torch.mean(x, 1)计算指定维度的平均值
class Residual_spa_att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.eca = eca_layer(channel=out_channels, k_size=5)

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(32, in_channels),
            # nn.SiLU(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(32, out_channels),
            # nn.SiLU(),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.relu = nn.ReLU()

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)

        x_in = h
        h = self.compress(h)
        h = self.spatial(h)
        h = self.sigmoid(h) * x_in
        return self.relu(h + self.skip(x))

class Residual_spa_ca_att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.eca = eca_layer(channel=out_channels, k_size=5)

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(32, in_channels),
            # nn.SiLU(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(32, out_channels),
            # nn.SiLU(),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.relu = nn.ReLU()

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)

        x_in = h
        h = self.compress(h)
        h = self.spatial(h)
        h = self.sigmoid(h) * x_in
        h = self.eca(h)
        return self.relu(h + self.skip(x))

# class Self_Attention_local(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, ph=4, pw=4, attn_drop=0., proj_drop=0.):
#         super(Self_Attention_local).__init__()
#         """
#         Args:
#             dim (_type_): the number of the channel dimension of the input feature map.
#             num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
#             qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
#             qk_scale (_type_, optional): Defaults to None.
#             attn_drop (_type_, optional): Defaults to 0..
#             proj_drop (_type_, optional): Defaults to 0..
#         """
#         self.ph = ph
#         self.pw = pw
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw) d', ph=self.ph, pw=self.pw)
#
#         B, R, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, R, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(-1, -2).reshape(B, R, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         x = rearrange(x, 'b (nh nw) (ph pw) d -> b d (nh ph) (nw pw)', nh=h // self.ph, nw=w // self.pw,
#                           ph=self.ph, pw=self.pw)
#         return x

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(320, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_3_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_3_4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x1_up = self.upsample(x1)
        x2_up = self.upsample(x2)

        x2_1 = x2 * x1_up
        x2_2 = self.conv_2_1(x2_1) + x1_up
        x2_3 = self.conv_2_2(x2_2)
        x2_3_up = self.upsample(x2_3)

        x3_1 = x3 * x2_up
        x3_2 = self.conv_3_1(x3_1) + x2_up
        x3_3 = self.conv_3_2(x3_2)
        x3_4 = x3_3 * x2_3_up
        x3_5 = self.conv_3_3(x3_4) + x2_3_up
        x3_5 = self.conv_3_4(x3_5)

        x = self.conv(x3_5)

        return x3_5, x

class region_fuse(nn.Module):
    def __init__(self, channel):
        super(region_fuse, self).__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(channel, 1, 1)

    def forward(self, x, x_boundary, x_boundary_map, x_saliency):
        x_saliency1 = x_saliency * x_boundary_map + x_saliency
        x_uncertain = (1 - torch.abs(x_saliency1.sigmoid() - 0.5) / 0.5)

        # x_saliency_sig = x_saliency.sigmoid()

        x = x * x_saliency + x
        x = self.conv1_1(x)

        x = x * x_boundary + x
        x = self.conv1_2(x)
        # x_uncertain = self.conv1_3(x_uncertain)
        # x = torch.cat((x_saliency, x_boundary), 1)
        # x = self.conv2(x)
        x = x_uncertain * x + x
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class Glob_loc_scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Glob_loc_scale, self).__init__()
        self.Res_att = Residual_spa_ca_att(out_channel, out_channel)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )


        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(),
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.dconv_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.dconv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.dconv_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.dconv_4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        # x = self.Res_att(x)
        x1 = self.conv0(x)
        left = x1
        right = x1

        # Multi-scale
        x1 = self.dconv_1(left)
        x2 = self.dconv_2(left) + x1
        x3 = self.dconv_3(left) + x2
        x4 = self.dconv_4(left) + x3
        x = torch.cat([x1, x2, x3, x4], 1)
        output = self.conv1(x)

        Hx = self.Hattn(right)
        Wx = self.Wattn(Hx)

        x = torch.cat([output, Wx], 1)
        output = self.conv2(x) + x1
        output = self.conv3(output)

        x_att = self.Res_att(output)
        output = self.conv4(x_att)
        return output

class Muti_Scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Muti_Scale, self).__init__()

        self.eca = eca_layer(out_channel, 5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, out_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel * 4),
            nn.ReLU(),
            nn.Conv2d(out_channel * 4, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = torch.cat([x1, x2, x3], 1)
        x1 = channel_shuffle(x, 32)

        output = self.conv5(x)
        output1 = self.conv4(x1)
        x = torch.cat([output, output1], 1)
        output = self.conv6(x)
        output = self.eca(output)

        return output

# class Muti_Scale(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Muti_Scale, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel // 4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel // 4),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#         self.dconv_1_1 = nn.Sequential(
#             nn.Conv2d(out_channel // 4, out_channel // 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_1_2 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_1_3 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channel // 8),
#             nn.ReLU()
#         )
#
#         self.dconv_2_1 = nn.Sequential(
#             nn.Conv2d(out_channel // 4, out_channel // 16, kernel_size=3, stride=1, padding=3, dilation=3),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_2_2 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 16, kernel_size=3, stride=1, padding=3, dilation=3),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_2_3 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 8, kernel_size=3, stride=1, padding=3, dilation=3),
#             nn.BatchNorm2d(out_channel // 8),
#             nn.ReLU()
#         )
#
#         self.dconv_3_1 = nn.Sequential(
#             nn.Conv2d(out_channel // 4, out_channel // 16, kernel_size=3, stride=1, padding=5, dilation=5),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_3_2 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 16, kernel_size=3, stride=1, padding=5, dilation=5),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_3_3 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 8, kernel_size=3, stride=1, padding=5, dilation=5),
#             nn.BatchNorm2d(out_channel // 8),
#             nn.ReLU()
#         )
#
#         self.dconv_4_1 = nn.Sequential(
#             nn.Conv2d(out_channel // 4, out_channel // 16, kernel_size=3, stride=1, padding=7, dilation=7),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_4_2 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 16, kernel_size=3, stride=1, padding=7, dilation=7),
#             nn.BatchNorm2d(out_channel // 16),
#             nn.ReLU()
#         )
#
#         self.dconv_4_3 = nn.Sequential(
#             nn.Conv2d(out_channel // 16, out_channel // 8, kernel_size=3, stride=1, padding=7, dilation=7),
#             nn.BatchNorm2d(out_channel // 8),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         input = x
#         x = self.conv2(x)
#
#         x1_1 = self.dconv_1_1(x)
#         x1_2 = self.dconv_1_2(x1_1)
#         x1_3 = self.dconv_1_3(x1_2)
#
#         x2_1 = self.dconv_2_1(x)
#         x2_2 = self.dconv_2_2(x2_1)
#         x2_3 = self.dconv_2_3(x2_2)
#
#         x3_1 = self.dconv_3_1(x)
#         x3_2 = self.dconv_3_2(x3_1)
#         x3_3 = self.dconv_3_3(x3_2)
#
#         x4_1 = self.dconv_4_1(x)
#         x4_2 = self.dconv_4_2(x4_1)
#         x4_3 = self.dconv_4_3(x4_2)
#
#         x1 = torch.cat([x1_1, x1_2, x1_3], 1)
#         x2 = torch.cat([x2_1, x2_2, x2_3], 1)
#         x3 = torch.cat([x3_1, x3_2, x3_3], 1)
#         x4 = torch.cat([x4_1, x4_2, x4_3], 1)
#
#         ad1 = x1
#         ad2 = ad1 + x2
#         ad3 = ad2 + x3
#         ad4 = ad3 + x4
#
#         output = torch.cat([ad1, ad2, ad3, ad4], 1)
#         output = self.conv3(output)
#         output = output + input
#
#         return output

# class Muti_Scale(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Muti_Scale, self).__init__()
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv1_0 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#         self.conv1_1 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv1_2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv1_3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv2_0 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3, dilation=3),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#         self.conv2_1 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3, dilation=3),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv2_2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=5, dilation=5),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#         self.conv2_3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=7, dilation=7),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(out_channel * 4, out_channel * 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel * 2),
#             nn.ReLU(),
#             nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.conv0(x)
#         input = x
#         x0 = self.conv1_0(x)
#         x1 = self.conv1_1(x + x0)
#         x2 = self.conv1_2(x + x1)
#         x3 = self.conv1_3(x + x2)
#
#         x0 = self.conv2_0(x0)
#         x1 = self.conv2_1(x1 + x0)
#         x2 = self.conv2_2(x2 + x1)
#         x3 = self.conv2_3(x3 + x2)
#
#         x = torch.cat([x0, x1, x2, x3], 1)
#
#         x = self.conv3(x)
#         x = torch.cat([x, input], 1)
#         x = self.conv4(x)
#
#         return x

class Boundary_Extract(nn.Module):
    def __init__(self):
        super(Boundary_Extract, self).__init__()

        self.Res_att = Residual_spa_att(64, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.conv7x7 = nn.Sequential(
            # DCN(32, 32, kernel_size=(7, 3), stride=1, padding=(3, 1), groups=32, bias=False),
            nn.Conv2d(32, 32, (7, 3), stride=1, padding=(3, 1), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # DCN(32, 32, kernel_size=(3, 7), stride=1, padding=(1, 3), groups=32, bias=False),
            nn.Conv2d(32, 32, (3, 7), stride=1, padding=(1, 3), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv11x11 = nn.Sequential(
            # DCN(32, 32, kernel_size=(11, 3), stride=1, padding=(5, 1), groups=32, bias=False),
            nn.Conv2d(32, 32, (11, 3), stride=1, padding=(5, 1), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # DCN(32, 32, kernel_size=(3, 11), stride=1, padding=(1, 5), groups=32, bias=False),
            nn.Conv2d(32, 32, (3, 11), stride=1, padding=(1, 5), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv17x17 = nn.Sequential(
            # DCN(32, 32, kernel_size=(17, 5), stride=1, padding=(8, 2), groups=32, bias=False),
            nn.Conv2d(32, 32, (17, 5), stride=1, padding=(8, 2), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # DCN(32, 32, kernel_size=(5, 17), stride=1, padding=(2, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (5, 17), stride=1, padding=(2, 8), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x, y, z):


        x = self.Res_att(x)
        x_out2 = torch.mean(x, 1).unsqueeze(1)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y_out0 = torch.mean(y, 1).unsqueeze(1)
        c = torch.cat([x, y], dim=1)
        fuse = self.conv1(c)
        fuse = self.conv2(fuse + x + y)
        fuse_out1 = torch.mean(fuse, 1).unsqueeze(1)
        fuse1 = self.conv7x7(fuse)
        fuse2 = self.conv11x11(fuse)
        fuse3 = self.conv17x17(fuse)

        fuse_cat = torch.cat([fuse1, fuse2, fuse3], dim=1)

        fuse = self.conv3(fuse_cat) + fuse
        fuse_out2 = torch.mean(fuse, 1).unsqueeze(1)
        fuse = fuse*(1-torch.sigmoid(z))
        fuse_out3 = torch.mean(fuse, 1).unsqueeze(1)
        fuse = self.conv4(fuse)
        boundary = self.conv5(fuse)

        return fuse, boundary, y_out0, x_out2, fuse_out1, fuse_out2, fuse_out3


class IGBP(nn.Module):
    def __init__(self, img_size=352, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False,
                 embed_dims=[64, 128, 256, 512]):
        super(IGBP, self).__init__()

        self.pvt = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )
        self.pvt.load_state_dict(torch.load('pretrained/pvt_v2_b3.pth'))

        self.MS4 = Muti_Scale(512, 32)
        self.Glo_loc3 = Glob_loc_scale(320, 32)
        self.Glo_loc2 = Glob_loc_scale(128, 32)

        self.agg = aggregation(32)

        self.fusion_4_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fusion_3_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fusion_2_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.boundary_extract = Boundary_Extract()

        self.region_fuse1 = region_fuse(32)
        self.region_fuse2 = region_fuse(32)
        self.region_fuse3 = region_fuse(32)

    def forward(self, imgs, labels=None):
        B = imgs.shape[0]

        x_att_4x, H, W = self.pvt.patch_embed1(imgs)
        for blk in self.pvt.block1:
            x_att_4x = blk(x_att_4x, H, W)
        x_att_4x = self.pvt.norm1(x_att_4x)
        x_att_4x = x_att_4x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_att_8x, H, W = self.pvt.patch_embed2(x_att_4x)
        for blk in self.pvt.block2:
            x_att_8x = blk(x_att_8x, H, W)
        x_att_8x = self.pvt.norm2(x_att_8x)
        x_att_8x = x_att_8x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_att_16x, H, W = self.pvt.patch_embed3(x_att_8x)
        for blk in self.pvt.block3:
            x_att_16x = blk(x_att_16x, H, W)
        x_att_16x = self.pvt.norm3(x_att_16x)
        x_att_16x = x_att_16x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_att_32x, H, W = self.pvt.patch_embed4(x_att_16x)
        for blk in self.pvt.block4:
            x_att_32x = blk(x_att_32x, H, W)
        x_att_32x = self.pvt.norm4(x_att_32x)
        x_att_32x = x_att_32x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        saliency, saliency_map = self.agg(x_att_32x, x_att_16x, x_att_8x)
        saliency_map = F.interpolate(saliency_map, scale_factor=2, mode='bilinear', align_corners=True)

        x_att_32x = self.MS4(x_att_32x)
        x_att_8x = self.Glo_loc2(x_att_8x)
        x_att_16x = self.Glo_loc3(x_att_16x)

        x_att_32x_up = F.interpolate(x_att_32x, scale_factor=2, mode='bilinear', align_corners=True)  # 16
        x_att_32x = self.fusion_4_3(x_att_32x_up)
        x_16x = torch.cat([x_att_16x, x_att_32x], dim=1)  # x_att_16x + x_att_32x
        x_16x_up = F.interpolate(x_16x, scale_factor=2, mode='bilinear', align_corners=True)  # 8
        x_16x = self.fusion_3_2(x_16x_up)
        x_8x = torch.cat([x_att_8x, x_16x], dim=1)  # x_att_8x + x_16x
        x_8x = self.fusion_2_1(x_8x)
        x_4x = x_att_4x
        # x_4x, x_boundary = self.boundary_extract(x_4x, x_8x)
        x_4x, x_boundary, out1, out2, out3, out4, out5 = self.boundary_extract(x_4x, saliency, saliency_map)

        x_att_32x = F.interpolate(x_att_32x, scale_factor=4, mode='bilinear', align_corners=True)  # 4
        x_32x = self.region_fuse3(x_att_32x, x_4x, x_boundary, saliency_map)

        x_att_16x = F.interpolate(x_16x, scale_factor=2, mode='bilinear', align_corners=True)  # 4
        x_16x = self.region_fuse2(x_att_16x, x_4x, x_boundary, x_32x)

        x_att_8x = F.interpolate(x_8x, scale_factor=2, mode='bilinear', align_corners=True)
        x_8x = self.region_fuse1(x_att_8x, x_4x, x_boundary, x_16x)

        x_boundary = F.interpolate(x_boundary, scale_factor=4, mode='bilinear', align_corners=True)
        f4 = F.interpolate(x_32x, scale_factor=4, mode='bilinear', align_corners=True)
        f3 = F.interpolate(x_16x, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(x_8x, scale_factor=4, mode='bilinear', align_corners=True)
        f_saliency = F.interpolate(saliency_map, scale_factor=4, mode='bilinear', align_corners=True)

        out1 = F.interpolate(out1, scale_factor=4, mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=True)
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, scale_factor=4, mode='bilinear', align_corners=True)
        out5 = F.interpolate(out5, scale_factor=4, mode='bilinear', align_corners=True)

        # x = torch.cat([f2, f3, f4], dim=1)
        # x = self.final(x)

        # return f2, out1, out2, out3, out4, out5, x_boundary
        return f2, f3, f4, f_saliency, x_boundary


