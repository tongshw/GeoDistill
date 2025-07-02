import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np


def generate_gaussian_heatmap(corr_dict, levels, sigma=1.0, gt=None):
    """
    生成一个以最大置信度点为中心的高斯分布 heatmap。

    Parameters:
    - heatmap (Tensor): 输入的热图，形状为 [B, H, W]
    - sigma (float): 高斯分布的标准差，控制分布的宽度

    Returns:
    - gaussian_heatmap (Tensor): 生成的高斯分布 heatmap，形状与输入 heatmap 相同
    """
    result_dict = {}
    for _, level in enumerate(levels):
        corr = corr_dict[level]
        corr = -(corr - 2) / 2
        B, H, W = corr.shape

        # 找到每个热图中的最大置信度点
        max_values, max_indices = torch.max(corr.view(B, -1), dim=1)

        if gt is None:
            max_y = max_indices // W
            max_x = max_indices % W
        else:
            scale = 64
            if level == 2:
                scale = 256
            gt_points = gt * scale / 4
            crop_x, crop_y = (scale - W) / 2, (scale - H) / 2
            gt_points[:, 0] = gt_points[:, 0] + scale/2 - crop_x
            gt_points[:, 1] = gt_points[:, 1] + scale/2 - crop_y
            max_y, max_x = gt_points[:, 1], gt_points[:, 0]

        # 创建一个新的 heatmap 以高斯分布的形式
        gaussian_heatmap = torch.zeros_like(corr)

        # 对每一个 batch，基于最大置信度点生成高斯分布
        for b in range(B):
            y, x = max_y[b].item(), max_x[b].item()

            # 使用 2D 高斯函数来创建分布
            Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            Y = Y.float().to(corr.device)
            X = X.float().to(corr.device)

            # 计算每个点到最大置信度点 (y, x) 的距离
            distance = (Y - y) ** 2 + (X - x) ** 2

            # 生成高斯分布
            gaussian_map = torch.exp(-distance / (2 * sigma ** 2))

            # gaussian_map = 2 - 2 * gaussian_map

            # 将其赋值给对应的 batch
            gaussian_heatmap[b] = gaussian_map

        result_dict[level] = gaussian_heatmap

    return result_dict

def cross_entropy_fully_supervised(pred_dict, target_dict, levels, s_temp=0.2, t_temp=0.09):
    ce_losses = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        pred = -(pred - 2) / 2

        b, h, w = pred.shape

        pred_map_flat = pred.view(b, -1)
        target_map_flat = target.view(b, -1)

        pred_map_softmax = F.softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)


        pred_map = pred_map_softmax.view(b, h, w)
        target_map = target_map_softmax.view(b, h, w)

        loss = -torch.sum(target_map * torch.log(pred_map)) / b
        ce_losses.append(loss)
    return torch.mean(torch.stack(ce_losses, dim=0).float())



def cross_entropy(pred_dict, target_dict, levels, s_temp=0.2, t_temp=0.09):
    ce_losses = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        pred = -(pred - 2) / 2
        target = -(target - 2) / 2

        b, h, w = pred.shape

        pred_map_flat = pred.view(b, -1)
        target_map_flat = target.view(b, -1)

        pred_map_softmax = F.softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)


        pred_map = pred_map_softmax.view(b, h, w)
        target_map = target_map_softmax.view(b, h, w)

        loss = -torch.sum(target_map * torch.log(pred_map)) / b
        ce_losses.append(loss)
    return torch.mean(torch.stack(ce_losses, dim=0).float())



def multi_scale_contrastive_loss(corr_maps, levels):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    '''
    matching_losses = []

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

    return torch.mean(torch.stack(matching_losses, dim=0))
