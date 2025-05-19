import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.nn.functional import interpolate


def L2_norm(x):
    B, C, H, W = x.shape
    y = F.normalize(x.reshape(B, C * H * W))
    return y.reshape(B, C, H, W)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 第一个批量归一化层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二个全连接层
        self.bn2 = nn.BatchNorm1d(output_size)  # 第二个批量归一化层

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        x = x.view(-1, x.size(2))  # 将 x 变形为 [batch_size * seq_length, feature_dim]

        out = self.fc1(x)  # 输入经过第一个全连接层
        out = self.bn1(out)  # 通过第一个批量归一化层
        out = self.relu(out)  # 通过 ReLU 激活函数
        out = self.fc2(out)  # 输入经过第二个全连接层
        out = self.bn2(out)  # 通过第二个批量归一化层

        out = out.view(batch_size, seq_length, -1)  # 将输出重新变形为 [batch_size, seq_length, output_dim]
        return out


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class DPT(nn.Module):
    def __init__(self, input_dims=[1536, 1536, 1536, 1536], output_dim=64, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, int(hidden_dim / 2), 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(int(hidden_dim / 2), int(hidden_dim / 2), 3, padding=1),
        )
        self.out_conv_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),  # 最终输出64通道
        )

        self.out_conv_3 = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, padding=1),  # 最终输出16通道
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4

        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)  # deep
        out1 = self.ref_2(feats[2], out)
        out2 = self.ref_1(feats[1], out1)
        out3 = self.ref_0(feats[0], out2)

        out1 = interpolate(out1, scale_factor=1)
        out1 = self.out_conv_1(out1)
        out2 = interpolate(out2, scale_factor=2)
        out2 = self.out_conv_2(out2)
        out3 = interpolate(out3, scale_factor=4)
        out3 = self.out_conv_3(out3)
        return [L2_norm(out1), L2_norm(out2), L2_norm(out3)]
