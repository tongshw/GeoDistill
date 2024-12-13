import torch
from torch import nn

from .efficientnet_pytorch.model import EfficientNet


class LocationPredictionNet(nn.Module):
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


    def forward(self, sat, bev):
        # 提取卫星图像特征
        sat_features = self.sat_efficientnet.extract_features(sat)

        # 处理地面图像特征（如果使用孪生网络）
        if self.grd_efficientnet is not None:
            bev_features = self.grd_efficientnet.extract_features(bev)

        else:
            bev_features = self.sat_efficientnet.extract_features(bev)


        return 0