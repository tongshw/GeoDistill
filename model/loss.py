import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np


def Weakly_supervised_loss_w_GPS_error(corr_maps, gt_shift_u, gt_shift_v, levels, meters_per_pixel, GPS_error=5):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    gt_shift_u: [B]
    gt_shift_v: [B]
    meters_per_pixel: [B], corresponding to original image size
    GPS_error: scalar, in terms of meters
    '''
    matching_losses = []

    # ---------- preparing for GPS error Loss -------
    # levels = [int(item) for item in args.level.split('_')]

    GPS_error_losses = [0]

    # ------------------------------------------------

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

    #     # ---------- preparing for GPS error Loss -------
    #     w = (torch.round(W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
    #     h = (torch.round(H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4)).long()    # [B]
    #     radius = (torch.ceil(GPS_error / (meters_per_pixel * np.power(2, 3 - level)))).long()
    #     GPS_dis = []
    #     for b_idx in range(M):
    #         # GPS_dis.append(torch.min(corr[b_idx, b_idx, h[b_idx]-radius: h[b_idx]+radius, w[b_idx]-radius: w[b_idx]+radius]))
    #         start_h = torch.max(torch.tensor(0).long(), h[b_idx] - radius[b_idx])
    #         end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b_idx] + radius[b_idx])
    #         start_w = torch.max(torch.tensor(0).long(), w[b_idx] - radius[b_idx])
    #         end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b_idx] + radius[b_idx])
    #         GPS_dis.append(torch.min(
    #             corr[b_idx, b_idx, start_h: end_h, start_w: end_w]))
    #     GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    # return torch.mean(torch.stack(matching_losses, dim=0)), torch.mean(torch.stack(GPS_error_losses, dim=0))
    return torch.mean(torch.stack(matching_losses, dim=0))

def consistency_constraint(corr_maps1, corr_maps2, levels):
    consistency_losses = []
    for _, level in enumerate(levels):
        corr1 = corr_maps1[level]
        corr2 = corr_maps2[level]
        B, B, H, W = corr1.shape
        pos_corr1 = corr1[torch.arange(B), torch.arange(B)]
        pos_corr2 = corr2[torch.arange(B), torch.arange(B)]

        min_indices1 = torch.argmin(pos_corr1.view(B, -1), dim=1)
        min_indices2 = torch.argmin(pos_corr2.view(B, -1), dim=1)

        rows1 = min_indices1 // W
        cols1 = min_indices1 % W

        rows2 = min_indices2 // W
        cols2 = min_indices2 % W

        distance_l1 = torch.abs(rows1 - rows2) + torch.abs(cols1 - cols2)
        consistency_losses.append(distance_l1)

    return torch.mean(torch.stack(consistency_losses, dim=0).float())


class MultiScaleLoss:
    def __init__(self, spatial_size=512, sigma=5):
        """
        多尺度GT生成器

        Args:
        - spatial_size: 原始空间维度
        - sigma: 高斯核标准差
        """
        self.spatial_size = spatial_size
        self.sigma = sigma
        self.l1loss = nn.L1Loss(reduction='mean')
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def generate_gaussian_gt(self, gt_points, spatial_size=None):
        """
        生成高斯平滑的二维GT

        Args:
        - gt_points: tensor, shape [batch_size, 2]
        - spatial_size: 可选的空间维度，默认使用初始化的维度

        Returns:
        - gt_heatmap: tensor, shape [batch_size, spatial_size, spatial_size]
        """
        spatial_size = spatial_size or self.spatial_size
        batch_size = gt_points.shape[0]
        gt_heatmap = torch.zeros(batch_size, spatial_size, spatial_size, device=gt_points.device)

        x = torch.arange(spatial_size, device=gt_points.device).float()
        y = torch.arange(spatial_size, device=gt_points.device).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        for b in range(batch_size):
            x0, y0 = gt_points[b][0], gt_points[b][1]
            gaussian = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * self.sigma ** 2))
            gt_heatmap[b] = gaussian

        # 修改归一化方式
        gt_heatmap = gt_heatmap / (gt_heatmap.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-8)
        return gt_heatmap


    def downsample_gt(self, gt_heatmap, target_size):
        """
        下采样GT热图

        Args:
        - gt_heatmap: tensor, shape [batch_size, original_size, original_size]
        - target_size: 目标下采样大小

        Returns:
        - downsampled_gt: tensor, shape [batch_size, target_size, target_size]
        """
        return F.interpolate(
            gt_heatmap.unsqueeze(1),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    import torch

    def sample_from_downsampled_gt(self, corr, downsampled_gt, offset=20):
        """
        根据corr坐标从下采样的GT中采样

        Args:
        - corr: tensor, shape [batch_size, corr_height, corr_width]
        - downsampled_gt: tensor, shape [batch_size, gt_size, gt_size]
        - offset: 坐标偏移量

        Returns:
        - sampled_gt: tensor, shape [batch_size, corr_height, corr_width]
        """
        batch_size, corr_height, corr_width = corr.shape
        gt_size = downsampled_gt.shape[1]
        sampled_gt = torch.zeros_like(corr)

        # 计算出所有corr坐标的偏移量
        coords_x = torch.arange(corr_width, device=corr.device).repeat(corr_height, 1) + offset
        coords_y = torch.arange(corr_height, device=corr.device).repeat(corr_width, 1).T + offset

        # 确保坐标在合法范围内
        coords_x = torch.clamp(coords_x, 0, gt_size - 1)
        coords_y = torch.clamp(coords_y, 0, gt_size - 1)

        # 计算邻近点的坐标
        x0 = torch.floor(coords_x).long()
        y0 = torch.floor(coords_y).long()
        x1 = torch.clamp(x0 + 1, 0, gt_size - 1)
        y1 = torch.clamp(y0 + 1, 0, gt_size - 1)

        # 计算插值权重
        wx = coords_x - x0.float()
        wy = coords_y - y0.float()

        # 使用下标进行批量索引，避免循环
        for b in range(batch_size):
            # 获取四个邻近点的值
            top_left = downsampled_gt[b, x0, y0]
            top_right = downsampled_gt[b, x1, y0]
            bottom_left = downsampled_gt[b, x0, y1]
            bottom_right = downsampled_gt[b, x1, y1]

            # 双线性插值计算
            sampled_value = (1 - wx) * (1 - wy) * top_left + \
                            wx * (1 - wy) * top_right + \
                            (1 - wx) * wy * bottom_left + \
                            wx * wy * bottom_right

            # 将插值结果放入 sampled_gt 中
            sampled_gt[b] = sampled_value

        return sampled_gt

    def l1_loss(self, sat_delta, predit_points):
        return self.l1loss(sat_delta, predit_points)

    def __call__(self, gt_points, corr_list, feat_list, kernel_list):
        """
        处理多尺度输入

        Args:
        - gt_points: tensor, shape [batch_size, num_points, 2]
        - corr_list: list of corr tensors, 每个tensor shape [batch_size, height, width]
        - kernel_list: 可选的kernel列表
        - offset: 坐标偏移量

        Returns:
        - loss_list: 每个尺度的损失列表
        - sampled_gt_list: 每个尺度采样的GT列表
        """
        # 生成原始空间的高斯GT
        gt_heatmap = self.generate_gaussian_gt(gt_points)

        loss_list = []
        sampled_gt_list = []

        for i in range(len(corr_list)):
            # 获取corr的尺度
            B, corr_height, corr_width = corr_list[i].shape

            # 下采样GT到对应尺度
            feat_size = feat_list[i].shape[2]
            downsampled_gt = self.downsample_gt(gt_heatmap, corr_width)
            downsampled_gt = downsampled_gt / downsampled_gt.sum(dim=(1, 2), keepdim=True)
            down_max = torch.max(downsampled_gt)

            # # 从下采样GT中采样
            # offset = kernel_list[i].shape[2] // 2
            # sampled_gt = self.sample_from_downsampled_gt(corr_list[i], downsampled_gt, offset)
            # sampled_gt = sampled_gt / sampled_gt.sum(dim=(1, 2), keepdim=True)
            #
            # sampled_gt_list.append(sampled_gt)
            # sample_max = torch.max(sampled_gt)

            # 计算损失
            # loss = self.BCELoss(corr_list[i], downsampled_gt)
            loss = self.cross_entropy_loss(corr_list[i].reshape(B, corr_height* corr_width), downsampled_gt.reshape(B, corr_height* corr_width))
            loss_list.append(loss)

        return torch.mean(torch.stack(loss_list))

