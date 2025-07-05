import torch
from torch import nn

from .efficientnet_pytorch.model import EfficientNet


class RotationPredictionNet(nn.Module):
    def __init__(self, args, num_classes=90):
        super().__init__()

        input_dim = 3

        self.sat_efficientnet = EfficientNet.from_pretrained(
            'efficientnet-b0',
            circular=False,
            in_channels=input_dim
        )

        self.grd_efficientnet = EfficientNet.from_pretrained(
            'efficientnet-b0',
            circular=False,
            in_channels=input_dim
        ) if args.p_siamese else None
        # self.grd_efficientnet = None


        self.sat_efficientnet._fc = nn.Identity()
        if self.grd_efficientnet:
            self.grd_efficientnet._fc = nn.Identity()

        feature_dim = 192
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(192 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, sat_img, grd_img=None):
        # sat_features = self.sat_efficientnet.extract_features(sat_img)
        if self.grd_efficientnet is not None:
            grd_feature_volume, multiscale_grd = self.grd_efficientnet.extract_features_multiscale(grd_img)
        else:
            grd_feature_volume, multiscale_grd = self.sat_efficientnet.extract_features_multiscale(grd_img)
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat_img)

        # grd_features = self.grd_efficientnet.extract_features(grd_img)

        combined_features = torch.cat([multiscale_sat[14], multiscale_grd[14]], dim=1)

        # batch_size = combined_features.size(0)  # 获取 batch size
        # combined_features = combined_features.view(batch_size, -1)

        # rotation_angle = self.fc(combined_features)

        fused_features = self.fusion_layer(combined_features)
        batch_size = fused_features.size(0)  # 获取 batch size
        fused_features = fused_features.view(batch_size, -1)

        rotation_angle = self.fc(fused_features)

        return rotation_angle