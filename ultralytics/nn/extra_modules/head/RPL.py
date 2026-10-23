import math
import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class RPLBoxBranch(nn.Module):
    """
    Residual Progressive Localization branch.

    Stage 1:
        Standard YOLO box feature extraction + coarse DFL prediction.

    Stage 2:
        Lightweight depthwise residual refinement for localization.

    Output:
        final_logits = coarse_logits + alpha * residual_logits
    """

    def __init__(self, c1, c2, reg_max=16):
        super().__init__()

        self.reg_max = reg_max

        # -------------------------------------------------
        # Stage 1: standard coarse localization
        # 与原 YOLO11 Detect.cv2 保持一致
        # -------------------------------------------------
        self.box_feat = nn.Sequential(
            Conv(c1, c2, 3),
            Conv(c2, c2, 3),
        )

        self.coarse_pred = nn.Conv2d(
            c2,
            4 * reg_max,
            kernel_size=1
        )

        # -------------------------------------------------
        # Stage 2: residual localization refinement
        #
        # DWConv:
        #   低成本提取局部空间/边界信息
        #
        # 1x1 Conv:
        #   重新进行通道组合
        # -------------------------------------------------
        self.refine_feat = nn.Sequential(
            DWConv(c2, c2, 3),
            Conv(c2, c2, 1),
        )

        self.refine_pred = nn.Conv2d(
            c2,
            4 * reg_max,
            kernel_size=1
        )

        # 每个检测尺度独立学习 residual 强度
        self.alpha = nn.Parameter(
            torch.tensor(1.0)
        )

        # -------------------------------------------------
        # 关键：
        # residual prediction 从 0 开始
        #
        # 初始：
        # final ≈ coarse
        #
        # 防止新分支一开始破坏原始定位结果
        # -------------------------------------------------
        nn.init.zeros_(self.refine_pred.weight)

        if self.refine_pred.bias is not None:
            nn.init.zeros_(self.refine_pred.bias)

    def forward(self, x):

        # Stage 1
        feat = self.box_feat(x)

        coarse = self.coarse_pred(feat)

        # Stage 2
        residual_feat = feat + self.refine_feat(feat)

        residual = self.refine_pred(residual_feat)

        # Progressive residual localization
        out = coarse + self.alpha * residual

        return out


class Detect_RPL(Detect):
    """
    YOLO11 detection head with
    Residual Progressive Localization (RPL).

    Classification branch:
        keep original YOLO Detect.cv3

    Regression branch:
        replace Detect.cv2 with RPLBoxBranch
    """

    def __init__(
        self,
        nc=80,
        reg_max=16,
        end2end=False,
        ch=...
    ):
        # -------------------------------------------------
        # 初始化原始 Detect
        # 保留：
        # cv3
        # DFL
        # stride
        # decode
        # inference
        # -------------------------------------------------
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch
        )

        # 与原 YOLO11 Detect 保持一致
        c2 = max(
            (
                16,
                ch[0] // 4,
                self.reg_max * 4
            )
        )

        # -------------------------------------------------
        # 只替换 bbox regression branch
        # classification cv3 完全不动
        # -------------------------------------------------
        self.cv2 = nn.ModuleList(
            RPLBoxBranch(
                c1=x,
                c2=c2,
                reg_max=self.reg_max
            )
            for x in ch
        )

        # YOLO end2end compatibility
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)

    def bias_init(self):
        """
        Initialize detection head biases.

        coarse_pred:
            使用 YOLO 原始 bbox bias

        refine_pred:
            bias = 0

        cls:
            完全使用原 Detect 初始化策略
        """

        for i, (box_head, cls_head) in enumerate(
            zip(
                self.one2many["box_head"],
                self.one2many["cls_head"]
            )
        ):

            # coarse regression
            box_head.coarse_pred.bias.data[:] = 2.0

            # residual regression starts from 0
            box_head.refine_pred.bias.data.zero_()

            # original YOLO classification initialization
            cls_head[-1].bias.data[: self.nc] = math.log(
                5
                / self.nc
                / (640 / self.stride[i]) ** 2
            )

        if self.end2end:

            for i, (box_head, cls_head) in enumerate(
                zip(
                    self.one2one["box_head"],
                    self.one2one["cls_head"]
                )
            ):

                box_head.coarse_pred.bias.data[:] = 2.0

                box_head.refine_pred.bias.data.zero_()

                cls_head[-1].bias.data[: self.nc] = math.log(
                    5
                    / self.nc
                    / (640 / self.stride[i]) ** 2
                )