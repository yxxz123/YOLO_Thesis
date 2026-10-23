"""
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-DOGEUCB.drawio
ultralytics/nn/module_images/自研模块-DOGEUCB.md.
"""

import warnings

warnings.filterwarnings("ignore")
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../..")
import torch
import torch.nn.functional as F
from calflops import calculate_flops
from torch import nn

from ultralytics.nn.modules.conv import Conv


class DynamicOffsetSampler(nn.Module):
    """动态偏移上采样器。借鉴 DySample 的坐标偏移思想：先用 1x1 卷积从输入特征预测 每组通道在 2x2 邻域内的可学习采样偏移，再用 grid_sample 对原始特征做内容自适应的 双线性重采样，替代固定的
    nearest 网格采样，缓解上采样带来的棋盘/锯齿伪影。.
    """

    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        assert in_channels >= groups and in_channels % groups == 0
        self.scale = scale
        self.groups = groups

        out_channels = 2 * groups * scale**2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)
        nn.init.constant_(self.scope.weight, 0)
        nn.init.constant_(self.scope.bias, 0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos.to(dtype=x.dtype)

        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device) + 0.5
        coords_w = torch.arange(w, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(dtype=x.dtype)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
            .reshape(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return (
            F.grid_sample(
                x.reshape(b * self.groups, -1, h, w).float(),
                coords.float(),
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            .reshape(b, -1, self.scale * h, self.scale * w)
            .to(dtype=x.dtype)
        )


class ChannelGate(nn.Module):
    """ECA 风格的轻量通道门控。对 channel_shuffle 后被重排、混合的通道做重要性重标定， 抑制打乱过程中被稀释/冗余的通道响应，避免信息在跨组重排后被均等对待。.
    """

    def __init__(self, k_size=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class DOGEUCB(nn.Module):
    """Dynamic-Offset Gated Efficient Up-Convolution Block (DOG-EUCB).

    在原始 EUCB 的基础上做两处改进：
    1. 将固定的 nearest 上采样替换为 DynamicOffsetSampler，根据输入内容预测逐组的
    亚像素采样偏移，用双线性重采样代替刚性网格采样，从源头缓解上采样锯齿伪影；
    2. 在 channel_shuffle 之后接入 ChannelGate，对被打乱重组的通道做重要性重标定，
    抑制冗余通道、增强关键通道，再送入逐点卷积做通道融合。
    """

    def __init__(self, in_channels, kernel_size=3, groups=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        groups = groups if in_channels % groups == 0 else 1
        self.dysample = DynamicOffsetSampler(in_channels, scale=2, groups=groups)
        self.dwc = Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels)
        self.gate = ChannelGate()
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.dysample(x)
        x = self.dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.gate(x)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


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
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = DOGEUCB(channel).to(device)

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
