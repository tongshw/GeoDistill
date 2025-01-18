import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np


def generate_gaussian_heatmap(corr_dict, levels, sigma=1.0):
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

        # 计算最大置信度点的 (y, x) 坐标
        max_y = max_indices // W
        max_x = max_indices % W

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


def cross_entropy(pred_dict, target_dict, levels, s_temp=0.2, t_temp=0.09):
    ce_losses = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        # target = target.detach()
        pred = -(pred - 2) / 2
        target = -(target - 2) / 2

        b, h, w = pred.shape
        # pred = pred[torch.arange(b), torch.arange(b)]
        # target = target[torch.arange(b), torch.arange(b)]

        pred_map_flat = pred.view(b, -1)
        target_map_flat = target.view(b, -1)

        pred_map_softmax = F.softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)
        # sum_pred = torch.sum(pred_map_softmax, dim=1)
        # max_pred = torch.max(pred_map_softmax, dim=1)[0]
        # sum_target = torch.sum(target_map_softmax, dim=1)
        # max_target = torch.max(target_map_softmax, dim=1)[0]
        #
        # bool_mask = target_map_softmax > 1e-2


        pred_map = pred_map_softmax.view(b, h, w)
        target_map = target_map_softmax.view(b, h, w)

        loss = -torch.sum(target_map * torch.log(pred_map)) / b
        ce_losses.append(loss)
    return torch.mean(torch.stack(ce_losses, dim=0).float())


def soft_argmax(corr_map):
    B, _, H, W = corr_map.shape
    # 创建坐标网格
    y_coords = torch.arange(0, H, dtype=torch.float32, device=corr_map.device).view(1, 1, H, 1).expand(B, 1, H, W)
    x_coords = torch.arange(0, W, dtype=torch.float32, device=corr_map.device).view(1, 1, 1, W).expand(B, 1, H, W)

    # 将 corr_map 扁平化并计算 softmax
    weights = F.softmax(corr_map.view(B, -1), dim=1).view(B, 1, H, W)

    # 计算加权平均位置（soft argmax）
    soft_x = (weights * x_coords).sum(dim=(2, 3))
    soft_y = (weights * y_coords).sum(dim=(2, 3))

    return soft_x, soft_y

def softmin(corr_map, tau=0.1):
    """
    对每个样本的相关图 (h, w) 维度应用 Softmin。
    Args:
        corr_map: 形状为 (batch, h, w) 的相关图
        tau: 温度参数，控制平滑程度
    Returns:
        softmin_probs: Softmin 后的分布，形状为 (batch, h, w)
    """
    batch, h, w = corr_map.shape

    # 将 (h, w) 展平成一维
    corr_map_flat = corr_map.view(batch, -1)

    # 应用 Softmin
    softmin_probs_flat = F.softmax(-corr_map_flat / tau, dim=-1)

    # 恢复到原来的形状
    softmin_probs = softmin_probs_flat.view(batch, h, w)

    return softmin_probs
def get_softmin_coordinates(softmin_probs):
    """
    根据 Softmin 概率分布计算平滑的最小值坐标。
    Args:
        softmin_probs: Softmin 概率分布，形状为 (batch, h, w)
    Returns:
        coordinates: 最小值的平滑坐标，形状为 (batch, 2)，即每个样本的 (x, y)
    """
    # 生成坐标网格
    batch, h, w = softmin_probs.shape
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # (h, w)

    # 将坐标移动到与 softmin_probs 相同的设备
    x_coords = x_coords.to(softmin_probs.device)
    y_coords = y_coords.to(softmin_probs.device)

    # 计算平滑的最小值坐标
    x_mean = torch.sum(softmin_probs * x_coords[None, :, :], dim=(-2, -1))  # (batch,)
    y_mean = torch.sum(softmin_probs * y_coords[None, :, :], dim=(-2, -1))  # (batch,)

    # 返回坐标 (x, y)
    return torch.stack([x_mean, y_mean], dim=-1)  # 形状为 (batch, 2)


def compute_kl_loss(corr_map1, corr_map2, temperature=0.1):
    epsilon = 1e-10  # 避免除零和 log(0)

    # 引入温度参数，调整分布
    corr_map1 = corr_map1 / temperature
    corr_map2 = corr_map2 / temperature

    # 步骤 1: 归一化为概率分布
    P = corr_map1 / (torch.sum(corr_map1, dim=(1, 2), keepdim=True) + epsilon)
    Q = corr_map2 / (torch.sum(corr_map2, dim=(1, 2), keepdim=True) + epsilon)

    # 步骤 2: 计算逐元素 KL 散度
    kl_div = P * torch.log(P / (Q + epsilon) + epsilon)
    kl_loss = torch.sum(kl_div, dim=(1, 2))  # 对 h 和 w 维度求和

    # 步骤 3: 平均 batch 的 KL 散度
    kl_loss = kl_loss.mean()
    return kl_loss



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

def consistency_constraint_soft_L1(corr_maps1, corr_maps2, levels):
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

        softmin_probs1 = softmin(pos_corr1, 0.001)
        max_s = softmin_probs1.max()
        softmin_probs2 = softmin(pos_corr2, 0.001)

        # 计算平滑最小值坐标
        coords1 = get_softmin_coordinates(softmin_probs1)  # 形状为 (batch, 2)
        coords2 = get_softmin_coordinates(softmin_probs2)  # 形状为 (batch, 2)

        # 计算欧几里得距离
        # l2_loss = torch.norm(coords1 - coords2, dim=-1)
        l1_loss = torch.sum(torch.abs(coords1 - coords2), dim=-1)

        # # 返回批量平均损失
        consistency_losses.append(l1_loss.mean())

    return torch.mean(torch.stack(consistency_losses, dim=0).float())

def consistency_constraint_KL_divergency(corr_maps1, corr_maps2, levels):
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

        kl_1 = compute_kl_loss(pos_corr1, pos_corr2)
        kl_2 = compute_kl_loss(pos_corr2, pos_corr1)
        # delta = pos_corr1 - pos_corr2
        #
        consistency_losses.append(kl_2.mean() + kl_1.mean())

    return torch.mean(torch.stack(consistency_losses, dim=0).float())
