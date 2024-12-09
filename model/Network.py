import torch
from torch import nn

from efficientnet_pytorch.model import EfficientNet


class RotationPredictionNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 定义输入通道数
        input_dim = 3  # 假设是RGB图像

        # 使用EfficientNet-B0作为特征提取器
        self.sat_efficientnet = EfficientNet.from_pretrained(
            'efficientnet-b0',
            circular=False,
            in_channels=input_dim
        )

        # 根据参数决定是否使用孪生网络结构
        self.grd_efficientnet = EfficientNet.from_pretrained(
            'efficientnet-b0',
            circular=False,
            in_channels=input_dim
        ) if args.p_siamese else None

        # 获取EfficientNet-B0的特征维度
        feature_dim = self.sat_efficientnet._fc.in_features

        # 移除原始分类器
        self.sat_efficientnet._fc = nn.Identity()
        if self.grd_efficientnet:
            self.grd_efficientnet._fc = nn.Identity()

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 旋转角度预测头
        self.rotation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # 预测单个旋转角度
        )

    def forward(self, sat_img, grd_img=None):
        # 提取卫星图像特征
        sat_features = self.sat_efficientnet.extract_features(sat_img)

        # 处理地面图像特征（如果使用孪生网络）
        if self.grd_efficientnet is not None and grd_img is not None:
            grd_features = self.grd_efficientnet.extract_features(grd_img)

            # 特征拼接
            combined_features = torch.cat([sat_features, grd_features], dim=1)

            # 特征融合
            fused_features = self.fusion_layer(combined_features)
        else:
            fused_features = sat_features

        # 预测旋转角度
        rotation_angle = self.rotation_head(fused_features)

        return rotation_angle