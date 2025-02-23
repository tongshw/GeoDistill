import os
import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import Normalize
import torch.nn.functional as F
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

class TextColors:
    RED = '31'
    GREEN = '32'
    YELLOW = '33'
    BLUE = '34'
    MAGENTA = '35'
    CYAN = '36'
    WHITE = '37'

def print_colored(text, color=TextColors.RED):
    print(f"\033[{color}m{text}\033[0m")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualization(bev, sat, sat_delta, ori_angle):
    batch_size = bev.shape[0]  # 批量大小为 2
    bev = bev.cpu().numpy()  # 转换为 NumPy
    sat = sat.cpu().numpy()  # 转换为 NumPy
    sat_delta = sat_delta.cpu().numpy()  # 假设形状为 (batch_size, 2)
    ori_angle = ori_angle.cpu().numpy()  # 假设形状为 (batch_size,)

    # 创建可视化
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))  # 每个样本两列

    # 遍历每个 batch 的数据
    for i in range(batch_size):
        print(sat_delta[i])
        # 处理 BEV 图像 (C, H, W) -> (H, W, C)
        bev_sample = bev[i].transpose(1, 2, 0)
        bev_sample = (bev_sample - bev_sample.min()) / (bev_sample.max() - bev_sample.min())  # 归一化

        # 处理 SAT 图像 (C, H, W) -> (H, W, C)
        sat_sample = sat[i].transpose(1, 2, 0)
        sat_sample = (sat_sample - sat_sample.min()) / (sat_sample.max() - sat_sample.min())  # 归一化

        # 绘制 BEV 图像
        axes[i, 0].imshow(bev_sample)
        axes[i, 0].set_title(f"BEV (ori_angle: {ori_angle[i]:.2f})", fontsize=12)
        axes[i, 0].axis("off")

        # 绘制 SAT 图像
        axes[i, 1].imshow(sat_sample)
        axes[i, 1].scatter(
            sat_delta[i][0],  # sat_delta_x 偏移
            sat_delta[i][1],  # sat_delta_y 偏移
            color="red",
            label="sat_delta"
        )
        axes[i, 1].set_title(f"SAT (ori_angle: {ori_angle[i]:.2f})", fontsize=12)
        axes[i, 1].legend()
        axes[i, 1].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()



def grid_sample(image, optical, jac=None):
    # values in optical within range of [0, H], and [0, W]
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(N, 1, H, W)
    iy = optical[..., 1].view(N, 1, H, W)

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x
        iy_nw = torch.floor(iy)  # north-west  upper-left-y
        ix_ne = ix_nw + 1        # north-east  upper-right-x
        iy_ne = iy_nw            # north-east  upper-right-y
        ix_sw = ix_nw            # south-west  lower-left-x
        iy_sw = iy_nw + 1        # south-west  lower-left-y
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0

    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:

        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # assert jac.shape[1:] == [N, H, W, 2]
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        if torch.any(torch.isnan(jac)) or torch.any(torch.isnan(dout_dpxy)):
            print('Nan occurs')

        return out_val, jac_new1 #jac_new1 #jac_new.permute(4, 0, 1, 2, 3)
    else:
        return out_val, None


def save_visualization(fig, save_path, dpi=300):
    """
    保存 matplotlib 图像到指定路径。
    如果路径不存在，则自动创建。

    Parameters:
        fig (matplotlib.figure.Figure): 要保存的 matplotlib 图像。
        save_path (str): 图像保存路径（包括文件名和后缀）。

    Returns:
        None
    """
    # 获取目录部分
    dir_path = os.path.dirname(save_path)

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")

    # 保存图像
    fig.savefig(save_path, format="jpeg", dpi=300, bbox_inches='tight')  # 保存为高分辨率图像
    # print(f"Figure saved to: {save_path}")

def map_corr_to_center_keep_sat(corr_map, sat_img, alpha=0):
    """
    将 155x155 的相关性图映射到 512x512 卫星图的中心区域，同时保留卫星图的外围区域。
    """
    # Step 1: 将 155x155 的相关性图放大到 310x310
    corr_map_resized = cv2.resize(corr_map.detach().cpu().numpy(), (310, 310), interpolation=cv2.INTER_CUBIC)
    corr_map_resized = Normalize(vmin=corr_map_resized.min(), vmax=corr_map_resized.max())(corr_map_resized)

    # Step 2: 将相关性图嵌入到 512x512 的中心
    corr_map_centered = np.zeros((512, 512), dtype=np.float32)  # 初始化为全零图像
    start_idx = (512 - 310) // 2  # 起始索引 (101, 101)
    corr_map_centered[start_idx:start_idx + 310, start_idx:start_idx + 310] = corr_map_resized

    # Step 3: 将相关性图转换为彩色
    cmap = plt.cm.Reds  # 红色渐变
    colored_corr_map = cmap(corr_map_centered)[:, :, :3]  # 去掉 Alpha 通道
    colored_corr_map = (colored_corr_map * 255).astype(np.uint8)  # 转换为 RGB 图像
    #
    alpha1 = (corr_map_centered * 255).astype(np.uint8)  # 值域 [0,255]

    # 合并 RGB 和 Alpha 通道为 RGBA
    rgba_image = np.zeros((512, 512, 4), dtype=np.uint8)  # 初始化全透明画布
    rgba_image[..., :3] = colored_corr_map  # 填充 RGB 颜色
    rgba_image[..., 3] = alpha1  # 填充 Alpha 通道

    # 保存为透明 PNG（注意边缘残留问题的处理）
    Image.fromarray(rgba_image, 'RGBA').save('corr_map_adaptive_alpha.png')

    # Step 4: 使用遮罩仅叠加中心区域
    sat_img_rgb = sat_img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)  # 卫星图转为 RGB
    mask = corr_map_centered > corr_map_centered.min()  # 创建遮罩，仅保留相关性图的有效区域
    for c in range(3):  # 对 RGB 通道分别处理
        sat_img_rgb[:, :, c] = np.where(mask,
                                        cv2.addWeighted(sat_img_rgb[:, :, c], 1 - alpha, colored_corr_map[:, :, c],
                                                        alpha, 0),
                                        sat_img_rgb[:, :, c])

    return sat_img_rgb

def vis_corr(corr_map, sat, bev, gt_point, pred_point, save_path=None, alpha=0.6, temp=1):
    # 处理相关性图（corr_map）
    if corr_map.max() > 1:
        corr_map = -(corr_map - 2) / 2

    h, w = corr_map.shape
    corr_map_flat = corr_map.view(-1)  # 展平成向量，形状为 (h*w,)

    # 对展平的相关性矩阵进行 softmax
    corr_map_softmax = F.softmax(corr_map_flat/temp, dim=0)  # 按所有元素求 softmax

    # 将 softmax 结果 reshape 回原来的 (h, w) 形状
    corr_map = corr_map_softmax.view(h, w)

    overlay = map_corr_to_center_keep_sat(corr_map, sat, alpha)

    # 创建画布
    fig = plt.figure(figsize=(16, 8))  # 根据需要调整画布大小

    # 使用 gridspec 定义布局
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 640 / 320])  # 调整宽高比例

    # 第一个子图：卫星图像和概率图叠加
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(overlay)  # Matplotlib 直接支持 RGB 图像
    if gt_point is not None:
        ax1.plot(gt_point[0].cpu().numpy(), gt_point[1].cpu().numpy(), marker='o', color='green', markersize=8,
                 label='GT')
        ax1.legend(loc='upper right')
    if pred_point is not None:
        ax1.plot(pred_point[0].cpu().numpy(), pred_point[1].cpu().numpy(), marker='d', color='blue', markersize=8,
                 label='pred')
        ax1.legend(loc='upper right')
    ax1.axis('on')
    ax1.set_title(f"Satellite Image with Probability Map(t={temp})")

    # 第二个子图：BEV图像（彩色 RGB）
    ax2 = fig.add_subplot(gs[1])
    _, h, w = bev.shape
    if h == w:
        bev = bev.permute(1, 2, 0)
    bev_img_rgb = bev.detach().cpu().numpy().astype(np.uint8)  # BEV 转换为 RGB
    ax2.imshow(bev_img_rgb)
    ax2.axis('on')
    ax2.set_title("BEV Image")

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        save_visualization(fig, save_path)
    else:
        plt.show()
    plt.close(fig)


def vis_two_sat(corr_map1, corr_map2, sat, bev, gt_point, pred_point1, pred_point2, save_path=None, alpha=0.6, temp=1):
    # 处理相关性图（corr_map）
    if corr_map1.min() >= 0:
        corr_map1 = -(corr_map1 - 2) / 2

    if corr_map2.min() >= 0:
        corr_map2 = -(corr_map2 - 2) / 2

    h, w = corr_map1.shape
    corr_map1_flat = corr_map1.view(-1)  # 展平成向量，形状为 (h*w,)
    corr_map2_flat = corr_map2.view(-1)  # 展平成向量，形状为 (h*w,)
    max_corr1, _ = torch.max(corr_map1_flat, dim=0)
    max_corr2, _ = torch.max(corr_map2_flat, dim=0)

    # 对展平的相关性矩阵进行 softmax
    corr_map1_softmax = F.softmax(corr_map1_flat/temp, dim=0)  # 按所有元素求 softmax
    corr_map2_softmax = F.softmax(corr_map2_flat / temp, dim=0)  # 按所有元素求 softmax

    # 将 softmax 结果 reshape 回原来的 (h, w) 形状
    corr_map1 = corr_map1_softmax.view(h, w)

    overlay1 = map_corr_to_center_keep_sat(corr_map1, sat, alpha)

    corr_map2 = corr_map2_softmax.view(h, w)

    overlay2 = map_corr_to_center_keep_sat(corr_map2, sat, alpha)

    # 创建画布
    fig = plt.figure(figsize=(20, 8))  # 根据需要调整画布大小

    # 使用 gridspec 定义布局
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 640 / 320])

    # 第一个子图：卫星图像和概率图叠加
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(overlay1)  # Matplotlib 直接支持 RGB 图像
    if gt_point is not None:
        ax1.plot(gt_point[0].cpu().numpy(), gt_point[1].cpu().numpy(), marker='^', color='blue', markersize=6,
                 label='GT')
        ax1.legend(loc='upper right')
    if pred_point1 is not None:
        ax1.plot(pred_point1[0].cpu().numpy(), pred_point1[1].cpu().numpy(), marker='*', color='green', markersize=12,
                 label=f"pred_before{max_corr1:.4f}")
        ax1.legend(loc='upper right')
    if pred_point2 is not None:
        ax1.plot(pred_point2[0].cpu().numpy(), pred_point2[1].cpu().numpy(), marker='*', color='yellow', markersize=12,
                 label=f"pred_after{max_corr2:.4f}")
        ax1.legend(loc='upper right')
    ax1.axis('on')
    ax1.set_title(f"before distillation")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(overlay2)  # 绘制卫星图像
    if gt_point is not None:
        ax2.plot(gt_point[0].cpu().numpy(), gt_point[1].cpu().numpy(), marker='^', color='blue', markersize=6,
                 label='GT')
        ax2.legend(loc='upper right')
    if pred_point2 is not None:
        ax2.plot(pred_point2[0].cpu().numpy(), pred_point2[1].cpu().numpy(), marker='*', color='yellow', markersize=12,
                 label='pred')
        ax2.legend(loc='upper right')
    ax2.axis('on')
    ax2.set_title("after distillation")

    # 第二个子图：BEV图像（彩色 RGB）
    ax3 = fig.add_subplot(gs[2])
    _, h, w = bev.shape
    if h == w:
        bev = bev.permute(1, 2, 0)
    bev_img_rgb = bev.detach().cpu().numpy().astype(np.uint8)  # BEV 转换为 RGB
    ax3.imshow(bev_img_rgb)
    ax3.axis('on')
    ax3.set_title("BEV Image")

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        save_visualization(fig, save_path)
    else:
        plt.show()
    plt.close(fig)


def visualize_distributions(distance_t, distance_s, output_dir, bins=None):
    """
    可视化两个数组的累计统计、区间统计（以百分比显示），以及它们差值的区间统计（以百分比显示），并将图片保存到本地。
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    if bins is None:
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 计算距离
    # distance_t = np.sqrt((pred_us_t - gt_us) ** 2 + (pred_vs_t - gt_vs) ** 2)  # [N]
    # distance_s = np.sqrt((pred_us_s - gt_us) ** 2 + (pred_vs_s - gt_vs) ** 2)  # [N]
    distance_diff = distance_t - distance_s  # 差值
    distance_diff = distance_diff[distance_s < distance_t]

    # 累计统计
    cumulative_edges = bins[1:]
    cumulative_counts_t = [np.sum(distance_t <= edge) for edge in cumulative_edges]
    cumulative_counts_s = [np.sum(distance_s <= edge) for edge in cumulative_edges]

    # 区间统计
    counts_t, _ = np.histogram(distance_t, bins=bins)
    counts_s, _ = np.histogram(distance_s, bins=bins)
    counts_diff, _ = np.histogram(distance_diff, bins=bins)

    # 计算总数
    total_count_t = len(distance_t)
    total_count_s = len(distance_s)
    total_count_diff = len(distance_diff)

    # 计算百分比
    percentages_t = counts_t / total_count_t * 100
    percentages_s = counts_s / total_count_s * 100
    percentages_diff = counts_diff / total_count_diff * 100

    # 绘制累计统计图
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_edges, cumulative_counts_t, label='Cumulative before', marker='o')
    plt.plot(cumulative_edges, cumulative_counts_s, label='Cumulative after', marker='o')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Cumulative Count')
    plt.title('Cumulative Statistics')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/cumulative_statistics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制区间统计图（百分比）
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], percentages_t, width=np.diff(bins), align='edge', alpha=0.7, label='Range before', edgecolor='black')
    plt.bar(bins[:-1], percentages_s, width=np.diff(bins), align='edge', alpha=0.7, label='Range after', edgecolor='black', color='orange')
    plt.xlabel('Distance Range')
    plt.ylabel('Percentage (%)')
    plt.title('Range Statistics (Percentage)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/range_statistics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制差值区间统计图（百分比）
    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], percentages_diff, width=np.diff(bins), align='edge', alpha=0.7, color='green', edgecolor='black', label='Difference Range')
    plt.xlabel('Distance Difference Range')
    plt.ylabel('Percentage (%)')
    plt.title('Difference Range Statistics (Percentage) after better than before')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/difference_statistics.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # print(f"Plots saved in {output_dir}")
