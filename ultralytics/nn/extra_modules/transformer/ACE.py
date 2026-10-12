"""
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/TCSVT2026-ACE.png
ultralytics/nn/module_images/TCSVT2026-ACE.md
论文链接：https://ieeexplore.ieee.org/document/11333296.
"""

import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from calflops import calculate_flops
from torch import nn


def build_norm_layer(norm_cfg, num_features):
    """Minimal replacement for mmcv.cnn.build_norm_layer supporting BN/SyncBN."""
    norm_type = norm_cfg.get("type", "BN") if isinstance(norm_cfg, dict) else norm_cfg
    if norm_type == "BN":
        return None, nn.BatchNorm2d(num_features)
    if norm_type == "SyncBN":
        return None, nn.SyncBatchNorm(num_features)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def constant_init(tensor, val):
    nn.init.constant_(tensor, val)


def trunc_normal_init(tensor, std=0.001):
    nn.init.trunc_normal_(tensor, std=std)


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, norm_cfg=None):
        if norm_cfg is None:
            norm_cfg = {"type": "BN", "requires_grad": True}
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class stripeconv(nn.Module):
    """
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8.
    """

    def __init__(self, in_channels, out_channels, inter_channels, num_heads=8):
        super().__init__()
        assert out_channels % num_heads == 0, (
            f"out_channels ({out_channels}) should be be a multiple of num_heads ({num_heads})"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=0.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=0.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.0)

    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1]
        )  # n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w).
        """
        x = self.norm(x)
        x1 = F.conv2d(x, self.kv, bias=None, stride=1, padding=(3, 0))
        x1 = self._act_dn(x1)
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1, padding=(3, 0))
        x3 = F.conv2d(x, self.kv3, bias=None, stride=1, padding=(0, 3))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0, 3))
        x = x1 + x3
        return x


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        # (B,C_qk,H)
        _B, _C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode="linear", align_corners=False)
        return x


class ACE(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        key_dim=None,
        attn_ratio=2,
        activation=nn.ReLU,
        norm_cfg=None,
    ):
        if norm_cfg is None:
            norm_cfg = {"type": "BN", "requires_grad": True}
        super().__init__()
        if key_dim is None:
            key_dim = dim // 4
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(
            2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1, groups=2 * self.dh, norm_cfg=norm_cfg
        )
        self.act = activation()

        self.cov = stripeconv(2 * self.dh, 2 * self.dh, inter_channels=64, num_heads=num_heads)
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)

    def forward(self, x):  # x (B,C,H,W)
        B, _C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Detail enhancement kernel
        qkv = torch.cat([q, k, v], dim=1)
        qkv_0 = qkv
        qkv = self.act(self.dwconv(qkv))
        qkv_0 = self.cov(qkv_0)
        qkv = qkv + qkv_0
        qkv = self.pwconv(qkv)

        # global attention
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        out_row = torch.matmul(attn_row, vrow)
        out_row = self.proj_encode_row(out_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        out_column = torch.matmul(attn_column, vcolumn)
        out_column = self.proj_encode_column(out_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        out = out_row.add(out_column)
        out = v.add(out)
        out = self.proj(out)
        out = out.sigmoid() * qkv
        return out


if __name__ == "__main__":
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = (
        "\033[91m",
        "\033[92m",
        "\033[94m",
        "\033[93m",
        "\033[38;5;208m",
        "\033[0m",
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = ACE(channel, num_heads=8).to(device)

    outputs = module(inputs)
    print(GREEN + f"inputs.size:{inputs.size()} outputs.size:{outputs.size()}" + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        input_shape=(batch_size, channel, height, width),
        output_as_string=True,
        output_precision=4,
        print_detailed=True,
    )
    print(RESET)

    half_outputs = module.half()(inputs.half())
    assert half_outputs.shape == outputs.shape and half_outputs.dtype is torch.float16
