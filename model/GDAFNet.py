# import paddle
# import cv2
# import os

# # # ==== 1. 加载现有模型 ====
# # from paddleseg.models.mynet_hr_re_lw import EnhancedHRNet_StripePooling_lw  # 你的模型类
# # model = EnhancedHRNet_StripePooling_lw()
# # for name, layer in model.named_sublayers():
# #     print(name)

# from paddleseg.models.EffiCrack_v4 import EffiCrack_v4

# model = EffiCrack_v4()

# with open('model_layers_detailed.txt', 'w', encoding='utf-8') as f:
#     for name, layer in model.named_sublayers():
#         # 写入层名称和层类型
#         f.write(f"{name}: {type(layer).__name__}\n")

# print("详细模型层信息已保存到 model_layers_detailed.txt 文件中")


import math
import numpy as np
import paddle
import paddle.nn as nn
from typing import Optional, Sequence
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
import paddle.nn.functional as F
from paddle.nn.initializer import Assign
from paddle.nn import Conv2D
from paddleseg.models.backbones.transformer_utils import (DropPath, ones_,
                                                          to_2tuple, zeros_)
from paddleseg.models import layers


class DAC(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, diagonal_type='main',
                 groups=1, bias_attr=False):
        super().__init__()
        self.diagonal_type = diagonal_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, (1, kernel_size),
                      padding=(0, kernel_size // 2), groups=groups, bias_attr=False),
            nn.BatchNorm2D(in_channels),
            nn.ReLU(),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=bias_attr)
        )
        

        self.cached_indices = {}
        self.cached_inverse_indices = {}

    def _compute_diagonal_indices_vectorized(self, H, W):
        i_coords = np.arange(H)[:, None]  # [H, 1]
        j_coords = np.arange(W)[None, :]  # [1, W]

        if self.diagonal_type == 'main':
            diag_idx = j_coords - i_coords + (H - 1)
        else:  
            diag_idx = i_coords + j_coords

        diag_idx_flat = diag_idx.flatten()  # [H*W]

        sort_order = np.argsort(diag_idx_flat)

        inverse_order = np.argsort(sort_order)

        return paddle.to_tensor(sort_order, dtype='int64'), paddle.to_tensor(inverse_order, dtype='int64')

    def _get_indices(self, H, W, device):
        key = (H, W, self.diagonal_type)

        if key not in self.cached_indices:
            indices, inverse_indices = self._compute_diagonal_indices_vectorized(H, W)
        
            if device is not None and hasattr(indices, 'place'):
                indices = indices._copy_to(device, False)
                inverse_indices = inverse_indices._copy_to(device, False)
            
            self.cached_indices[key] = indices
            self.cached_inverse_indices[key] = inverse_indices

        return self.cached_indices[key], self.cached_inverse_indices[key]

    def forward(self, x):
        N, C, H, W = x.shape
        device = x.place

        perm, inv_perm = self._get_indices(H, W, device)

        x_view = x.reshape([N, C, H * W])  # [N, C, H*W]
        x_rearranged = paddle.index_select(x_view, perm, axis=2)  # [N, C, H*W]
        x_rearranged = x_rearranged.reshape([N, C, H, W])

        diag_feat = self.conv(x_rearranged)

        diag_feat_view = diag_feat.reshape([N, self.out_channels, H * W])
        diag_feat_restored = paddle.index_select(diag_feat_view, inv_perm, axis=2)
        diag_feat_restored = diag_feat_restored.reshape([N, self.out_channels, H, W])


        return diag_feat_restored


class DSSE(nn.Layer):
    def __init__(self, c_r, out_channels, strip_scales, direction):
        super().__init__()

        half_ch = out_channels // 2
        remaining_ch = out_channels - half_ch

        if direction == 'horizontal':
            self.branch1 = nn.Sequential(
                nn.Conv2D(c_r, c_r, (1, strip_scales[0]), padding=(0, strip_scales[0] // 2),
                          groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, half_ch, 1, bias_attr=False)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2D(c_r, c_r, (1, strip_scales[1]), padding=(0, strip_scales[1] // 2),
                          groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, remaining_ch, 1, bias_attr=False)
            )
        elif direction == 'vertical':
            self.branch1 = nn.Sequential(
                nn.Conv2D(c_r, c_r, (strip_scales[0], 1), padding=(strip_scales[0] // 2, 0),
                          groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, half_ch, 1, bias_attr=False)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2D(c_r, c_r, (strip_scales[1], 1), padding=(strip_scales[1] // 2, 0),
                          groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, remaining_ch, 1, bias_attr=False)
            )
        elif direction == 'main_diagonal':
            self.branch1 = nn.Sequential(
                DAC(c_r, c_r, strip_scales[0], diagonal_type='main',
                                   groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, half_ch, 1, bias_attr=False)
            )
            self.branch2 = nn.Sequential(
                DAC(c_r, c_r, strip_scales[1], diagonal_type='main',
                                   groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, remaining_ch, 1, bias_attr=False)
            )
        else:  
            self.branch1 = nn.Sequential(
                DAC(c_r, c_r, strip_scales[0], diagonal_type='anti',
                                   groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, half_ch, 1, bias_attr=False)
            )
            self.branch2 = nn.Sequential(
                DAC(c_r, c_r, strip_scales[1], diagonal_type='anti',
                                   groups=c_r, bias_attr=False),
                nn.BatchNorm2D(c_r),
                nn.ReLU(),
                nn.Conv2D(c_r, remaining_ch, 1, bias_attr=False)
            )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        return paddle.concat([feat1, feat2], axis=1)


class MDPM(nn.Layer):
    def __init__(self, c_r, c_cheap, M=4, strip_scales=[3, 7]):
        super().__init__()
        self.c_r = c_r
        self.c_cheap = c_cheap
        self.M = M
        self.strip_scales = strip_scales

        self.strip_generators = nn.LayerList()

        direction_names = ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal']

        cheap_per_gen = max(1, c_cheap // M)
        remaining_cheap = c_cheap

        for i in range(M):
            if i == M - 1:  
                current_cheap = remaining_cheap
            else:
                current_cheap = cheap_per_gen
                remaining_cheap -= current_cheap

            if current_cheap <= 0:
                self.strip_generators.append(nn.Identity())
                continue

            generator = DSSE(
                c_r, current_cheap, strip_scales, direction=direction_names[i]
            )

            self.strip_generators.append(generator)

        self.direction_router = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(c_r, max(8, c_r // 4), 1),  
            nn.ReLU(),
            nn.Conv2D(max(8, c_r // 4), M, 1),
            nn.Sigmoid()
        )

    def forward(self, primary_feat):
        routing_weights = self.direction_router(primary_feat)  # [N, M, 1, 1]
        routing_weights = routing_weights.reshape([routing_weights.shape[0], self.M])  # [N, M]

        cheap_parts = []
        for i, generator in enumerate(self.strip_generators):
            if isinstance(generator, nn.Identity):
                continue

            cheap_part = generator(primary_feat)  # [N, cheap_channels, H, W]

            weight = routing_weights[:, i:i + 1].unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1, 1]
            weighted_part = cheap_part * weight
            cheap_parts.append(weighted_part)

        if len(cheap_parts) > 0:
            cheap_features = paddle.concat(cheap_parts, axis=1)  # [N, c_cheap, H, W]
        else:
            cheap_features = paddle.zeros([primary_feat.shape[0], self.c_cheap,
                                           primary_feat.shape[2], primary_feat.shape[3]],
                                          dtype=primary_feat.dtype)

        return cheap_features


class PCAF(nn.Layer):
    def __init__(self, primary_channels, cheap_channels, out_channels,
                 reduction=8, activation=nn.ReLU):
        super().__init__()
        self.primary_channels = primary_channels
        self.cheap_channels = cheap_channels
        self.total_channels = primary_channels + cheap_channels
        self.out_channels = out_channels

        self.primary_ca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(primary_channels, max(1, primary_channels // reduction), 1),
            activation(),
            nn.Conv2D(max(1, primary_channels // reduction), primary_channels, 1),
            nn.Sigmoid()
        )

        self.cheap_ca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(cheap_channels, max(1, cheap_channels // reduction), 1),
            activation(),
            nn.Conv2D(max(1, cheap_channels // reduction), cheap_channels, 1),
            nn.Sigmoid()
        )


        self.final_fusion = nn.Sequential(
            nn.Conv2D(self.total_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            activation()
        )

    def forward(self, primary_feat, cheap_feat):
        primary_attention = self.primary_ca(primary_feat)  # [N, primary_ch, 1, 1]
        cheap_attention = self.cheap_ca(cheap_feat)  # [N, cheap_ch, 1, 1]

        enhanced_primary = primary_feat * primary_attention 
        enhanced_cheap = cheap_feat * cheap_attention  

        concat_feat = paddle.concat([enhanced_primary, enhanced_cheap], axis=1)  # [N, total_ch, H, W]
        fused = self.final_fusion(concat_feat)

        return fused


class GDAFBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio=0.5,
                 M=4,  # number of strip generators
                 strip_scales=[3, 7],  
                 reduction=8,
                 activation=nn.ReLU,
                 residual_scale_init=0.0
                 ):
        super().__init__()
        assert 0.0 < ratio <= 1.0
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.ratio = ratio
        self.c_r = int(out_channels * ratio)  
        self.c_cheap = out_channels - self.c_r
        self.M = M
        self.strip_scales = strip_scales

        self.primary_pw = nn.Conv2D(in_channels, self.c_r, kernel_size=1, bias_attr=False)
        self.bn_primary = nn.BatchNorm2D(self.c_r)
        self.act = activation()

        self.strip_cheap_gen = MDPM(
            self.c_r, self.c_cheap, M, strip_scales
        )

        self.fuse = PCAF(
            primary_channels=self.c_r,
            cheap_channels=self.c_cheap,
            out_channels=out_channels,
            reduction=max(4, (self.c_r + self.c_cheap) // 8),
            activation=activation
        )
        self.apa_branch = AGA(in_channels, out_channels)

    def forward(self, x):
        y_r = self.primary_pw(x)  # [N, c_r, H, W]
        y_r = self.bn_primary(y_r)
        y_r = self.act(y_r)

        y_c = self.strip_cheap_gen(y_r)  # [N, c_cheap, H, W]
        y = self.fuse(y_r, y_c)  # [N, out_ch, H, W]

        apa_out = self.apa_branch(x)  # [N, out_ch, H, W]

        out =  y + apa_out

        return out


class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias_attr=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_depthwise_conv(dim, kernel_size=3):
    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    padding = tuple([k // 2 for k in kernel_size])
    return Conv2D(
        dim, dim, kernel_size, padding=padding, bias_attr=True, groups=dim)


class AGA(nn.Layer):
    def __init__(self, in_channel, out_channel=None, x=256):
        super(AGA, self).__init__()

        if out_channel is None:
            out_channel = in_channel
        self.pool_h_max = nn.AdaptiveMaxPool2D((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2D((1, None))

        self.max_h = nn.Sequential(
            nn.Conv1D(in_channel, in_channel, kernel_size=5, padding=2, groups=in_channel),
            nn.Conv1D(in_channel, out_channel, kernel_size=1, padding=0),
            nn.BatchNorm1D(in_channel)
        )
        self.max_w = nn.Sequential(
            nn.Conv1D(in_channel, in_channel, kernel_size=5, padding=2, groups=in_channel),
            nn.Conv1D(in_channel, out_channel, kernel_size=1, padding=0),
            nn.BatchNorm1D(in_channel)
        )

    def forward(self, x):
        x_h = self.pool_h_max(x).squeeze(axis=3)  # [1, 32, 100, 1]-->  [1, 32, 100]
        x_w = self.pool_w_max(x).squeeze(axis=2)
        out_h10 = self.max_h(x_h)
        out_w10 = self.max_w(x_w)
        a_w = paddle.nn.functional.sigmoid(out_w10).unsqueeze(axis=2)  # B N 1 W
        a_h = paddle.nn.functional.sigmoid(out_h10).unsqueeze(axis=3)  # B N H 1
        out = x * a_w * a_h
        return out

        
class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = get_depthwise_conv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Layer):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 kernel_sizes: Sequence[int] = (3, 3, 3),
                 dilations: Sequence[int] = (1, 1, 1),
                 expansion: float = 1.0,
                 ):
        super(DWConv, self).__init__()
        hidden_channels = (int)(out_channels * expansion)

        self.dw_conv = BasicConv(hidden_channels, hidden_channels, kernel_sizes[0], 1, kernel_sizes[0] // 2,
                                 dilations[0], groups=hidden_channels, bn=None, relu=None)

        self.pw_conv = BasicConv(hidden_channels, hidden_channels, 1, 1, 0,
                                 bn=True, relu=True)  

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = self.pw_conv(x)  
        return x + shortcut


class GDB(nn.Layer):
    def __init__(self,
                 hidden_channels: int = 256,
                 kernel_sizes: Sequence[int] = (3, 5, 7),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1),  
                 ffn_scale: float = 2.0,
                 dropout_rate: float = 0.,
                 drop_path_rate: float = 0.0,
                 ):
        super(GDB, self).__init__()

        self.norm1 = nn.BatchNorm2D(hidden_channels)

        strip_scales = [kernel_sizes[0], kernel_sizes[-1]] if len(kernel_sizes) >= 2 else [3, 7]

        self.block = GDAFBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            ratio=0.5,
            M=4,
            strip_scales=strip_scales,
            reduction=8,
            activation=nn.ReLU
        )

        self.norm2 = nn.BatchNorm2D(hidden_channels)
        self.mlp = Mlp(in_features=hidden_channels,
                       hidden_features=int(hidden_channels * ffn_scale),
                       drop=drop_path_rate)

        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        layer_scale_init_value = paddle.full(
            [hidden_channels, 1, 1], fill_value=1e-2, dtype="float32")
        self.layer_scale_1 = paddle.create_parameter(
            [hidden_channels, 1, 1], "float32", attr=Assign(layer_scale_init_value))
        self.layer_scale_2 = paddle.create_parameter(
            [hidden_channels, 1, 1], "float32", attr=Assign(layer_scale_init_value))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.block(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class DownSampleConv(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(DownSampleConv, self).__init__()
        self.branch_1 = nn.Sequential(
            BasicConv(in_planes, out_planes, 3, stride=2, padding=1, relu=True, bn=True)
        )

    def forward(self, x):
        return self.branch_1(x)


class Stem(nn.Layer):
    """Stem layer"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(Stem, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(out_channels // 2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2D(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2D(out_channels // 2),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2D(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv1_3(self.conv1_2(self.conv1_1(x)))


class GDCEncoder(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, block_num: int, stage_num: int = 2,
                 drop_path: float = 0.1, kernel_sizes: Sequence[int] = (3, 5, 7)):
        super(GDCEncoder, self).__init__()
        self.stage_num = stage_num
        self.down_sample = DownSampleConv(in_channels, out_channels)
        self.blocks = nn.Sequential(*[
            GDB(out_channels, drop_path_rate=drop_path, kernel_sizes=kernel_sizes)
            for _ in range(block_num)
        ])

    def forward(self, x):
        if self.stage_num != 1:
            x = self.down_sample(x)  
        return self.blocks(x)  



class UpsamplingBottleneck(nn.Layer):
    def __init__(
            self,
            scale_factor: int
    ):
        super(UpsamplingBottleneck, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
    def forward(self, x):
        return self.up(x)


class CSA(nn.Layer):
    def __init__(self, dim_q, dim_kv, num_heads=4):
        super(CSA, self).__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.head_dim = dim_q // num_heads

        assert dim_q % num_heads == 0

        self.q_proj = nn.Sequential(
            BasicConv(dim_q, dim_q, 3, stride=1, padding=1, groups=dim_q, relu=True, bn=True),
            BasicConv(dim_q, dim_q, 1, stride=1, padding=0, relu=None, bn=None)
        )
        self.k_proj = nn.Sequential(
            BasicConv(dim_kv, dim_kv, 3, stride=1, padding=1, groups=dim_kv, relu=True, bn=True),
            BasicConv(dim_kv, dim_q, 1, stride=1, padding=0, relu=None, bn=None)
        )
        self.v_proj = nn.Sequential(
            BasicConv(dim_kv, dim_kv, 3, stride=1, padding=1, groups=dim_kv, relu=True, bn=True),
            BasicConv(dim_kv, dim_q, 1, stride=1, padding=0, relu=None, bn=None)
        )
        self.out_proj = nn.Sequential(
            BasicConv(dim_q, dim_q, 3, stride=1, padding=1, groups=dim_q, relu=True, bn=True),
            BasicConv(dim_q, dim_q, 1, stride=1, padding=0, relu=None, bn=None)
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, q, kv):
        Q = self.q_proj(q)  # [N, C_q, H_q, W_q]
        K = self.k_proj(kv)  # [N, C_q, H_kv, W_kv]
        V = self.v_proj(kv)  # [N, C_q, H_kv, W_kv]

        out = F.sigmoid(Q * K * self.scale) * V  # [N, C_q, H_kv, W_kv]


        out = self.out_proj(out)
        return out


class HCAFDecoder(nn.Layer):
    def __init__(self, feat_channels: Sequence[int]):
        super(HCAFDecoder, self).__init__()
        self.feat_channels = feat_channels

        self.fine3 = nn.Sequential(
            BasicConv(feat_channels[2], feat_channels[2], 3, stride=1, padding=1, groups=feat_channels[2], relu=None,
                      bn=None),
            BasicConv(feat_channels[2], feat_channels[1], 1, stride=1, padding=0, relu=True, bn=True),
            BasicConv(feat_channels[1], feat_channels[1], 3, stride=1, padding=1, groups=feat_channels[1], relu=None,
                      bn=None),
            BasicConv(feat_channels[1], feat_channels[1], 1, stride=1, padding=0, relu=True, bn=True),
        )
        self.conv3 = BasicConv(feat_channels[2], feat_channels[1], 1, stride=1, padding=0, relu=True, bn=True)

        self.fine2 = nn.Sequential(
            BasicConv(feat_channels[1], feat_channels[1], 3, stride=1, padding=1, groups=feat_channels[1], relu=None,
                      bn=None),
            BasicConv(feat_channels[1], feat_channels[1], 1, stride=1, padding=0, relu=True, bn=True)
        )
        self.conv2 = BasicConv(feat_channels[1], feat_channels[0], 1, stride=1, padding=0, relu=True, bn=True)
        
        self.shuffle = nn.ChannelShuffle(feat_channels[1] + feat_channels[0])
        self.fine0 = nn.Sequential(
            BasicConv(feat_channels[1] + feat_channels[0], feat_channels[1] + feat_channels[0], 3, stride=1, padding=1,
                      groups=feat_channels[1] + feat_channels[0], relu=None, bn=None),
            BasicConv(feat_channels[1] + feat_channels[0], feat_channels[1] + feat_channels[0], 1, stride=1, padding=0,
                      relu=True, bn=True),
            BasicConv(feat_channels[1] + feat_channels[0], feat_channels[1] + feat_channels[0], 3, stride=1, padding=1,
                      groups=feat_channels[1] + feat_channels[0], relu=None, bn=None),
            BasicConv(feat_channels[1] + feat_channels[0], feat_channels[1] + feat_channels[0], 1, stride=1, padding=0,
                      relu=True, bn=True),
        )
        self.att0 = AGA(feat_channels[1]+feat_channels[0], feat_channels[1]+feat_channels[0])

        self.concat = nn.Sequential(
            BasicConv(feat_channels[1]+feat_channels[0], feat_channels[1], kernel_size=1, groups=1, bn=True, relu=True),
            BasicConv(feat_channels[1], feat_channels[1], 3, 1, 1, groups=feat_channels[1], bn=None, relu=None),
            BasicConv(feat_channels[1], feat_channels[1], kernel_size=1, groups=1, bn=True, relu=True)
        )

        self.cross_attn_1to2 = CSA(dim_q=feat_channels[0], dim_kv=feat_channels[1], num_heads=4)
        self.cross_attn_2to1 = CSA(dim_q=feat_channels[1], dim_kv=feat_channels[1], num_heads=4)

        self.unshuffle = nn.PixelUnshuffle(2)
        self.unshuffle_proj = nn.Conv2D(feat_channels[0] * 4, feat_channels[1], 1)

        self.proj_1to2 = nn.Conv2D(feat_channels[0], feat_channels[0], 1)
        self.proj_2to1 = nn.Conv2D(feat_channels[1], feat_channels[0], 1)

    def forward(self, feat1, feat2, feat3, h, w):
        feat2_up = F.interpolate(feat2, size=(paddle.shape(feat1)[2], paddle.shape(feat1)[3]),
                                 mode='bilinear', align_corners=True)
        cross_1to2 = self.cross_attn_1to2(feat1, feat2_up)
        cross_1to2 = self.proj_1to2(cross_1to2)

        feat1_unshuffled = self.unshuffle(feat1)
        feat1_unshuffled = self.unshuffle_proj(feat1_unshuffled)
        cross_2to1 = self.cross_attn_2to1(feat2, feat1_unshuffled)
        cross_2to1_up = F.interpolate(cross_2to1, size=(paddle.shape(feat1)[2], paddle.shape(feat1)[3]),
                                      mode='bilinear', align_corners=True)
        cross_2to1_up = self.proj_2to1(cross_2to1_up)

        aux_cross = cross_1to2 + cross_2to1_up

        feat3_0 = self.conv3(feat3)
        feat3_1 = self.fine3(feat3)
        aux_feat3 = F.interpolate((feat3_0 + feat3_1), size=(h // 4, w // 4), mode='bilinear',
                              align_corners=True)

        out = self.shuffle(paddle.concat([aux_cross, aux_feat3], axis=1))
        out0 = self.att0(out)
        out1 = self.fine0(out)
        out = out0 + out1
        out = self.concat(out)

        return out, aux_cross, aux_feat3


class GDAFNet(nn.Layer):
    def __init__(
            self,
            base: int,
            block_num: Sequence[int] = (2, 2, 4),
            feat_channels: Sequence[int] = (32, 64, 128),
            num_classes=2,  # number of classes
            pretrained=None,  # pretrained model
            dropout_rate: float = 0.1
    ):
        super(GDAFNet, self).__init__()
        self.pretrained = pretrained
        self.feat_channels = feat_channels
        self.stem = Stem(3, base)
        self.stage1 = GDCEncoder(feat_channels[0], feat_channels[0], block_num[0], stage_num=1, drop_path=dropout_rate,
                           kernel_sizes=(3, 7))  
        self.stage2 = GDCEncoder(feat_channels[0], feat_channels[1], block_num[1], drop_path=dropout_rate,
                           kernel_sizes=(3, 7))  
        self.stage3 = GDCEncoder(feat_channels[1], feat_channels[2], block_num[2], drop_path=dropout_rate,
                           kernel_sizes=(5, 9))

        self.decoder = HCAFDecoder(feat_channels)

        self.act = nn.ReLU()
        self.segHead = BasicConv(feat_channels[1], num_classes, kernel_size=1, groups=1, bn=None, relu=None)
        self.segHead_edge = BasicConv(feat_channels[1], num_classes, kernel_size=1, groups=1, bn=None, relu=None)

        self.aux_edge1 = BasicConv(feat_channels[0], num_classes, kernel_size=1, groups=1, bn=None, relu=None)
        self.aux_edge2 = BasicConv(feat_channels[1], num_classes, kernel_size=1, groups=1, bn=None, relu=None)

        self.aux_seg1 = BasicConv(feat_channels[0], num_classes, kernel_size=1, groups=1, bn=None, relu=None)
        self.aux_seg2 = BasicConv(feat_channels[1], num_classes, kernel_size=1, groups=1, bn=None, relu=None)
        self.init_weight()

    def forward(self, x):
        h, w = paddle.shape(x)[2:]
        logit_list = []
        x = self.stem(x)

        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)

        out, aux_cross, aux_feat3 = self.decoder(feat1, feat2, feat3, h, w)
        seg = self.segHead(out)
        edg_m = self.segHead_edge(out) * paddle.nn.functional.sigmoid(seg)
        edg_m = F.interpolate(edg_m, size=(h, w), mode='bilinear', align_corners=True)

        seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=True) + self.act(edg_m)
        if self.training:
            seg1 = self.aux_seg1(aux_cross)
            edg1 = self.aux_edge1(aux_cross) * paddle.nn.functional.sigmoid(seg1)
            edg1 = F.interpolate(edg1, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
            seg1 = F.interpolate(seg1, size=(h // 2, w // 2), mode='bilinear', align_corners=True) + self.act(edg1)

            seg2 = self.aux_seg2(aux_feat3)
            edg2 = self.aux_edge2(aux_feat3) * paddle.nn.functional.sigmoid(seg2)
            edg2 = F.interpolate(edg2, size=(h//2, w//2), mode='bilinear', align_corners=True)
            seg2 = F.interpolate(seg2, size=(h//2, w//2), mode='bilinear', align_corners=True)  + self.act(edg2)

            logit_list = [seg, edg_m, seg1, edg1,seg2, edg2]
        else:
            logit_list = [seg]
        logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)


@manager.MODELS.add_component
def GDAFNet_B(num_classes=2):
    model = GDAFNet(32, (1, 1, 2), (32, 64, 128), dropout_rate=0.1, pretrained=None)
    return model


if __name__ == "__main__":
    model = GDAFNet(32, (1, 1, 2), (32, 64, 128))
    x = paddle.randn([1, 3, 400, 400])
    out = model(x)
    print(out[0].shape)
    model.eval()
    paddle.flops(model, input_size=(1, 3, 400, 400))




    

