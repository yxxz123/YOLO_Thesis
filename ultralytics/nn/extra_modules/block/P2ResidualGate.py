
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class P2ResidualGate(nn.Module):
    """
    P2 Residual Gated Fusion.

    将P2浅层细节特征投影到P3主干特征空间，
    通过可学习的通道级残差门控进行融合。

    Args:
        c_main: P3主干特征通道数
        c_bridge: P2 Bridge输入通道数
        init_alpha: 残差门控初始值，默认0.1

    Inputs:
        [main, bridge]

    Output:
        与main具有相同形状的融合特征
    """

    def __init__(
        self,
        c_main: int,
        c_bridge: int,
        init_alpha: float = 0.1,
    ):
        super().__init__()

        if not 0.0 < float(init_alpha) < 1.0:
            raise ValueError(
                f"init_alpha must be between 0 and 1, got {init_alpha}"
            )

        # P2特征独立投影，不参与P3三路Concat
        self.bridge_proj = Conv(
            c_bridge,
            c_main,
            k=1,
            s=1,
            act=False
        )

        # 将初始门控值转换为Logit
        init_logit = math.log(
            float(init_alpha) / (1.0 - float(init_alpha))
        )

        # 通道级可学习门控
        self.gate_logit = nn.Parameter(
            torch.full(
                (1, c_main, 1, 1),
                init_logit,
                dtype=torch.float32
            )
        )

    @property
    def alpha(self):
        """返回经过Sigmoid约束的门控系数。"""
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):

        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "P2ResidualGate expects [main, bridge] as inputs."
            )

        main, bridge = x

        # P2浅层特征通道投影
        bridge = self.bridge_proj(bridge)

        # 保证空间尺寸一致
        if bridge.shape[-2:] != main.shape[-2:]:
            bridge = F.interpolate(
                bridge,
                size=main.shape[-2:],
                mode="nearest"
            )

        # 通道级残差门控
        alpha = self.alpha.to(dtype=main.dtype)

        # 残差注入
        out = main + alpha * bridge

        return out
