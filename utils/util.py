import os
import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import Normalize
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA


CameraGPS_shift = [1.08, 0.26]
Satmap_zoom = 18
Camera_height = 1.65 #meter
Camera_distance = 0.54 #meter

SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 256 # 0.2 m per pixel
Default_lat = 49.015

CameraGPS_shift_left = [1.08, 0.26]
CameraGPS_shift_right = [1.08, 0.8]  # 0.26 + 0.54
def get_process_satmap_sidelength():
    return SatMap_process_sidelength


def gps2meters_torch(lat_s, lon_s, lat_d=torch.tensor([49.015]), lon_d=torch.tensor([8.43])):
    # inputs: torch array: [n]
    r = 6378137  # equatorial radius
    flatten = 1 / 298257  # flattening
    E2 = flatten * (2 - flatten)
    m = r * np.pi / 180
    lat = lat_d[0]
    coslat = np.cos(lat * np.pi / 180)
    w2 = 1 / (1 - E2 * (1 - coslat * coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1 - E2)

    x = (lon_d - lon_s) * kx
    y = (lat_s - lat_d) * ky  # y: from top to bottom

    return x, y


def gps2distance(lat_s, lon_s, lat_d, lon_d ):
    x,y = gps2meters_torch(lat_s, lon_s, lat_d, lon_d )
    dis = torch.sqrt(torch.pow(x, 2)+torch.pow(y,2))
    return dis


def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)
    meter_per_pixel /= 2 # because use scale 2 to get satmap
    meter_per_pixel /= scale
    return meter_per_pixel

def get_camera_height():
    return Camera_height


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


def visualize_feature_map_pca(feature_map, index=0):
    """
    使用 PCA 对 feature map 进行降维并可视化。
    :param feature_map: 形状为 (B, C, H, W) 的张量
    :param index: 选择 batch 维度中的哪一个样本进行可视化
    """
    assert feature_map.dim() == 4, "Feature map should have shape (B, C, H, W)"
    feature = feature_map[index]  # 选择 batch 内的某个样本, 形状 (C, H, W)
    C, H, W = feature.shape

    # 将 (C, H, W) 转换为 (H*W, C) 以便 PCA 处理
    feature_reshaped = feature.view(C, -1).T  # 变为 (H*W, C)

    # 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    feature_pca = pca.fit_transform(feature_reshaped)  # (H*W, 3)

    # 归一化到 [0, 1] 以便可视化
    feature_pca -= feature_pca.min()
    feature_pca /= feature_pca.max()

    # 重新 reshape 回 (H, W, 3)
    feature_pca_image = feature_pca.reshape(H, W, 3)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_pca_image)
    plt.axis('off')
    plt.title("Feature Map Visualization via PCA")
    plt.show()


def visualize_feature_map(feature_map, original_image=None, save_path=None):
    """
    可视化特征图

    参数:
        feature_map: 特征图张量
        original_image: 原始图像 (可选)，用于叠加热力图
        save_path: 保存可视化结果的路径 (可选)
    """
    # 确保输入是numpy数组
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    # 检查并调整维度顺序
    if len(feature_map.shape) == 4:
        # 如果是[B,C,H,W]
        feature_map = feature_map[0]  # 只取第一个样本

    if feature_map.shape[0] < feature_map.shape[1] and feature_map.shape[0] < feature_map.shape[2]:
        # 如果是 [C,H,W] 格式
        attention_map = np.mean(feature_map, axis=0)
    else:
        # 如果是 [H,W,C] 格式
        attention_map = np.mean(feature_map, axis=-1)

    # 归一化到0-1范围
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    # 确保是单通道8位图像
    attention_map_uint8 = np.uint8(255 * attention_map)

    # 生成热力图
    heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_JET)

    # 如果提供了原始图像，则叠加热力图到原图
    if original_image is not None:
        # 处理原始图像
        if isinstance(original_image, str):
            # 如果是图像路径
            original_img = Image.open(original_image).convert('RGB')
            original_img = np.array(original_img)
        elif isinstance(original_image, np.ndarray):
            # 如果已经是numpy数组
            original_img = original_image.copy()
        elif isinstance(original_image, torch.Tensor):
            # 如果是torch张量
            original_img = original_image.detach().cpu().numpy()
            if len(original_img.shape) == 4:
                # 如果是[B,C,H,W]格式
                original_img = original_img[0]  # 取第一个样本

            if original_img.shape[0] == 3 and len(original_img.shape) == 3:
                # 如果是[C,H,W]格式，转换为[H,W,C]
                original_img = np.transpose(original_img, (1, 2, 0))

        # 确保像素值在0-1之间或0-255之间
        if original_img.max() <= 1.0:
            # 如果是0-1范围，转换为0-255
            original_img_display = (original_img * 255).astype(np.uint8)
        else:
            # 如果已经是0-255范围
            original_img_display = original_img.astype(np.uint8)

        # 调整原图大小以匹配热力图
        original_img_resized = cv2.resize(original_img_display, (attention_map.shape[1], attention_map.shape[0]))

        # 确保原图是RGB顺序(而不是BGR)
        if len(original_img_resized.shape) == 3 and original_img_resized.shape[2] == 3:
            # 对于matplotlib显示，确保是RGB顺序
            original_img_rgb = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB)
        else:
            original_img_rgb = original_img_resized

        # 叠加热力图到原图 (OpenCV使用BGR顺序)
        superimposed_img = cv2.addWeighted(original_img_resized, 0.6, heatmap, 0.4, 0)

        # 显示原图和热力图叠加结果
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original_img_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(attention_map, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # 转换为RGB顺序用于matplotlib显示
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(superimposed_img_rgb)
        plt.title('Overlay')
        plt.axis('off')
    else:
        # 如果没有原始图像，只显示热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_map, cmap='jet')
        plt.title('Feature Map Attention')
        plt.colorbar()
        plt.axis('off')

    plt.tight_layout()

    # 保存结果
    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存到 {save_path}")

    plt.show()

    return attention_map



def generate_MAE_mask(image, mask_ratio=0.75, patch_size=16):
    """
    生成与输入图片形状相同的mask，随机将部分16x16区域置0（覆盖75%面积），其余置1。

    参数：
    image: numpy数组，输入图片，形状为(H, W)或(H, W, C)。
    mask_ratio: float，需覆盖的面积比例，默认为0.75。
    patch_size: int，每个区域的大小，默认为16。

    返回：
    mask: numpy数组，与image形状相同的0-1矩阵。
    """
    h, w = image.shape[:2]
    target_area = mask_ratio * h * w  # 计算需要覆盖的总面积

    # 生成所有可能的块
    blocks = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            y_end = min(i + patch_size, h)
            x_end = min(j + patch_size, w)
            block_area = (y_end - i) * (x_end - j)
            blocks.append((i, j, y_end, x_end, block_area))

    # 随机打乱块顺序
    np.random.shuffle(blocks)

    # 选择块直到覆盖足够面积
    masked_area = 0
    selected_blocks = []
    for block in blocks:
        if masked_area >= target_area:
            break
        selected_blocks.append(block)
        masked_area += block[4]

    # 创建全1的mask
    mask = np.ones((h, w), dtype=np.uint8)

    # 将选中的块置0
    for y_start, x_start, y_end, x_end, _ in selected_blocks:
        mask[y_start:y_end, x_start:x_end] = 0

    # 扩展mask维度以匹配输入图片的形状（若为多通道）
    # if len(image.shape) == 3:
    #     mask = mask[:, :, np.newaxis].repeat(image.shape[2], axis=2)

    return mask



def generate_mask_avg(feature_map, r):
    """
    生成一个只包含0和1的mask,值为1的元素个数/所有元素比例为r。

    参数:
    feature_map (torch.Tensor): 输入的feature map, shape为(batch, channel, height, width)
    r (float): 需要保留的比例, 范围为(0, 1)

    返回:
    mask (torch.Tensor): 生成的mask, shape为(batch, 1, height, width)
    """
    batch_size, channel, height, width = feature_map.shape

    # 对通道维度进行平均池化 (使用均值池化)
    avg_pool = feature_map.mean(dim=1, keepdim=True)

    # 将池化后的结果按升序排序
    flat_avg_pool = avg_pool.view(batch_size, -1)  # 展平为(batch, height * width)
    k = int(r * height * width)
    _, indices = torch.topk(flat_avg_pool, k, dim=1, largest=True)

    # 生成 mask
    mask = torch.ones_like(avg_pool)
    mask.view(batch_size, -1).scatter_(1, indices, 0)

    return mask
