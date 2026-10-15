"""使用 PyTorch 基础组件手写 ResNet-18，不导入现成模型或预训练参数。

本文件供其他脚本导入；在项目根目录运行：
    .venv/Scripts/python.exe -m unittest discover -s tests -p test_model.py -v
模型输入为浮点 NCHW 图像，输出为每个类别的原始分数（logits）。
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """两层 3×3 卷积的残差块：输入 (N,Cin,H,W)，输出 (N,Cout,H',W')。

    stride=1 保留空间尺寸；stride=2 在第一层卷积降采样。
    只有通道和尺寸都相同时才直接跳跃，否则用 1×1 卷积投影以便相加。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or stride not in (1, 2):
            raise ValueError("通道数必须为正，残差块 stride 只能是 1 或 2。")
        # 后接 BatchNorm 会学习平移量，因此卷积无需再添加重复的 bias。
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 ReLU(F(x)+shortcut(x))；两条支路必须得到完全相同的形状。"""
        identity = self.shortcut(x)
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        # 跳跃连接让信息和梯度有直接通路；不能用拼接替代这里的逐元素相加。
        return self.relu(residual + identity)


class ResNet18(nn.Module):
    """四组 [2,2,2,2] BasicBlock，默认输出五类花卉的 logits。

    num_classes 决定最后一层输出数，必须与数据中的类别映射数量一致。
    18 层按惯例计数为：首层卷积 + 8×2 个主支路卷积 + 全连接层；不计投影。
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes 必须为正整数。")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(64, 64, stride=1)
        self.layer2 = self._make_stage(64, 128, stride=2)
        self.layer3 = self._make_stage(128, 256, stride=2)
        self.layer4 = self._make_stage(256, 512, stride=2)
        # 自适应平均池化消除空间维度，使全连接层只依赖通道数。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """每组固定两个块：仅第一个块允许更改通道或降采样。"""
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def _initialize_weights(self) -> None:
        """从随机参数开始：卷积采用适合 ReLU 的 Kaiming 初始化，不加载外部权重。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (N,3,H,W) 图像转换成 (N,num_classes) 原始分数。"""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("输入应为 (批量, 3, 高, 宽) 的 RGB 图像张量。")
        # 以 224×224 输入为例，注释列出各组输出的 NCHW 形状。
        x = self.stem(x)     # (N,64,56,56)，首层卷积后先为 (N,64,112,112)
        x = self.layer1(x)   # (N,64,56,56)
        x = self.layer2(x)   # (N,128,28,28)
        x = self.layer3(x)   # (N,256,14,14)
        x = self.layer4(x)   # (N,512,7,7)
        x = self.avgpool(x)  # (N,512,1,1)
        x = torch.flatten(x, 1)  # 保留批量轴：(N,512)
        # 不在这里做 Softmax：后续 CrossEntropyLoss 直接接收 logits 更稳定。
        return self.fc(x)
