import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models

from model.efficientnet_pytorch import EfficientNet


class ResNetBackbone(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # 分阶段特征提取
        self.early_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 特征调整
        self.feature_adjust = nn.Conv2d(2048, feature_dim, kernel_size=1)

    def forward(self, x):
        # 逐层特征提取
        x = self.early_features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通道调整
        x = self.feature_adjust(x)

        return x
class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels, grid_size=16):
        super().__init__()

        # 逐步上采样的反卷积层
        self.upsample = nn.Sequential(
            # 第一次上采样：4x
            nn.ConvTranspose2d(in_channels, in_channels // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),

            # 第二次上采样：8x
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),

            # 第三次上采样：16x
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),

            # 最终上采样到全分辨率
            nn.ConvTranspose2d(in_channels // 8, 2,
                               kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, bev_feat, sat_feat):
        B, C, H, W = bev_feat.shape

        # 将特征图重塑为序列
        bev_feat = bev_feat.flatten(2).transpose(1, 2)  # B, HW, C
        sat_feat = sat_feat.flatten(2).transpose(1, 2)  # B, HW, C

        # 计算attention
        q = self.query(bev_feat)  # B, HW, C
        k = self.key(sat_feat)  # B, HW, C
        v = self.value(sat_feat)  # B, HW, C

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # B, HW, C
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class AttentionGridRegistrationNet(nn.Module):
    def __init__(self, args, grid_size=16):
        super().__init__()

        # 保持原有的EfficientNet特征提取器
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

        self.sat_efficientnet._fc = nn.Identity()
        if self.grd_efficientnet:
            self.grd_efficientnet._fc = nn.Identity()

        feature_dim = 320

        # 添加CrossAttention
        self.cross_attention = CrossAttention(feature_dim)

        # 特征融合改进
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        # 网格分类分支改进
        self.grid_cls_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, grid_size ** 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(grid_size ** 2, grid_size ** 2),
            nn.Softmax(dim=1)
        )

        # 坐标回归分支改进
        self.coord_reg_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, 2, 1)
        )

        self.feature_upsampler = FeatureUpsampler(feature_dim, grid_size)

        # 注意力机制改进
        self.attention_weights = nn.Sequential(
            nn.Conv2d(grid_size ** 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.Sigmoid()
        )

        self.grid_size = grid_size

    def forward(self, sat_img, bev_img):
        # 图像预处理
        sat_img = 2 * (sat_img / 255.0) - 1.0
        bev_img = 2 * (bev_img / 255.0) - 1.0
        sat_img = sat_img.contiguous()
        bev_img = bev_img.contiguous()

        # 特征提取
        _, multiscale_grd = self.grd_efficientnet.extract_features_multiscale(bev_img)
        _, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat_img)

        grd_feat = multiscale_grd[15]
        sat_feat = multiscale_sat[15]

        # 应用CrossAttention
        attended_feat = self.cross_attention(grd_feat, sat_feat)

        # # 特征融合
        # fused_features = self.feature_fusion(
        #     torch.cat([attended_feat, grd_feat], dim=1)
        # )

        # 网格分类
        grid_cls = self.grid_cls_branch(attended_feat)

        # 注意力加权
        attention_mask = self.attention_weights(
            grid_cls.view(grid_cls.size(0), -1, 1, 1)
        )
        weighted_features = attended_feat * attention_mask

        coord_offset = self.coord_reg_branch(weighted_features)  # (B, 2, 16, 16)

        # 获取最可能的网格位置
        grid_cls_flat = grid_cls.view(-1, self.grid_size, self.grid_size)  # (B, 16, 16)
        max_prob_idx = torch.argmax(grid_cls_flat.view(grid_cls_flat.size(0), -1), dim=1)  # (B,)

        # 计算选中网格的坐标
        grid_y = max_prob_idx // self.grid_size
        grid_x = max_prob_idx % self.grid_size

        # 获取对应网格位置的偏移量
        batch_indices = torch.arange(coord_offset.size(0)).to(coord_offset.device)
        selected_offsets = coord_offset[batch_indices, :, grid_y, grid_x]  # (B, 2)

        # 计算网格大小
        cell_size = 512 // self.grid_size  # 假设输入图像大小为512

        # 计算网格中心坐标
        grid_center_x = (grid_x.float() + 0.5) * cell_size
        grid_center_y = (grid_y.float() + 0.5) * cell_size

        # 将偏移量转换为实际坐标（偏移量范围从[-1,1]转换到[-cell_size/2, cell_size/2]）
        final_x = grid_center_x + selected_offsets[:, 0] * (cell_size / 2)
        final_y = grid_center_y + selected_offsets[:, 1] * (cell_size / 2)

        # 组合最终预测的坐标
        pred_coords = torch.stack([final_x, final_y], dim=1)  # (B, 2)

        return grid_cls, pred_coords

        return grid_cls, coord_offset

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GridRegistrationLoss(nn.Module):
    def __init__(self, grid_size=8, img_size=512, cls_weight=10, reg_weight=1.0, sigma=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.img_size = img_size
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.sigma = sigma  # 高斯分布的标准差

    def calculate_grid_index(self, pixel_coords):
        grid_width = self.img_size // self.grid_size
        grid_x = int(pixel_coords[0] // grid_width)
        grid_y = int(pixel_coords[1] // grid_width)
        return grid_x, grid_y

    def gaussian_label(self, grid_x, grid_y):
        """
        生成高斯分布标签。
        """
        grid_indices = torch.zeros((self.grid_size, self.grid_size))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
                grid_indices[y, x] = math.exp(-distance / (2 * self.sigma ** 2))
        grid_indices /= grid_indices.sum()  # 归一化，使概率和为1
        return grid_indices.view(-1)

    def get_predicted_coords(self, full_res_pred):
        batch_size, _, height, width = full_res_pred.size()
        pred_x = full_res_pred[:, 0]  # (B, H, W)
        pred_y = full_res_pred[:, 1]  # (B, H, W)

        pred_coords = []
        for i in range(batch_size):
            x_max_idx = pred_x[i].argmax()
            y_max_idx = pred_y[i].argmax()

            x_coord = x_max_idx % width
            y_coord = y_max_idx // width

            pred_coords.append(torch.tensor([x_coord, y_coord]))

        return torch.stack(pred_coords).to(full_res_pred.device)

    def forward(self, pred_cls, coord_offset, gt_coords):
        batch_size = gt_coords.size(0)

        # 生成高斯平滑标签
        smooth_labels = []
        for i in range(batch_size):
            grid_x, grid_y = self.calculate_grid_index(gt_coords[i])
            smooth_label = self.gaussian_label(grid_x, grid_y).to(pred_cls.device)
            smooth_labels.append(smooth_label)

        smooth_labels = torch.stack(smooth_labels)  # (B, grid_size^2)

        # 分类损失 (基于高斯平滑标签的交叉熵)
        pred_cls_probs = F.log_softmax(pred_cls, dim=1)  # (B, grid_size^2)
        cls_loss = -torch.sum(smooth_labels * pred_cls_probs) / batch_size
        cls_loss *= self.cls_weight

        # 回归损失
        pred_coords = coord_offset.squeeze(-1).squeeze(-1)  # (B, 2)
        reg_loss = F.smooth_l1_loss(pred_coords, gt_coords) * self.reg_weight

        total_loss = cls_loss + reg_loss

        return total_loss


# 训练器
# class Trainer:
#     def __init__(self, grid_size=16, img_size=512):
#         self.model = AttentionGridRegistrationNet(grid_size)
#         self.criterion = GridRegistrationLoss(grid_size, img_size)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
#
#     def train_step(self, sat_img, bev_img, gt_coords):
#         # 清空梯度
#         self.optimizer.zero_grad()
#
#         # 前向传播
#         pred_cls, pred_reg = self.model(sat_img, bev_img)
#
#         # 计算损失
#         loss = self.criterion(pred_cls, pred_reg, gt_coords)
#
#         # 反向传播
#         loss.backward()
#
#         # 参数更新
#         self.optimizer.step()
#
#         return loss.item()
#
#     def train(self, dataloader, epochs=30):
#         for epoch in range(epochs):
#             total_loss = 0
#             for sat_img, bev_img, gt_coords in dataloader:
#                 loss = self.train_step(sat_img, bev_img, gt_coords)
#                 total_loss += loss
#
#             print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader)}")
#
#
#
# # 使用示例
# def main():
#     # 假设的数据加载器
#     class DummyDataLoader:
#         def __init__(self):
#             self.data = [
#                 (torch.randn(1, 3, 512, 512),  # sat_img
#                  torch.randn(1, 3, 512, 512),  # bev_img
#                  torch.tensor([256, 256]))  # gt_coords
#                 for _ in range(100)
#             ]
#
#         def __iter__(self):
#             return iter(self.data)
#
#         def __len__(self):
#             return len(self.data)
#
#     # 创建训练器
#     trainer = Trainer()
#
#     # 使用虚拟数据训练
#     dataloader = DummyDataLoader()
#     trainer.train(dataloader)
#
#
# if __name__ == "__main__":
#     main()
