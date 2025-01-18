import numpy as np
import torch
import torch.nn.functional as F


def generate_gaussian_heatmap(heatmap, sigma=1.0):
    """
    生成一个以最大置信度点为中心的高斯分布 heatmap。

    Parameters:
    - heatmap (Tensor): 输入的热图，形状为 [B, H, W]
    - sigma (float): 高斯分布的标准差，控制分布的宽度

    Returns:
    - gaussian_heatmap (Tensor): 生成的高斯分布 heatmap，形状与输入 heatmap 相同
    """
    # 获取 batch_size, height 和 width
    B, H, W = heatmap.shape

    # 找到每个热图中的最大置信度点
    max_values, max_indices = torch.max(heatmap.view(B, -1), dim=1)

    # 计算最大置信度点的 (y, x) 坐标
    max_y = max_indices // W
    max_x = max_indices % W

    # 创建一个新的 heatmap 以高斯分布的形式
    gaussian_heatmap = torch.zeros_like(heatmap)

    # 对每一个 batch，基于最大置信度点生成高斯分布
    for b in range(B):
        y, x = max_y[b].item(), max_x[b].item()

        # 使用 2D 高斯函数来创建分布
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        Y = Y.float().to(heatmap.device)
        X = X.float().to(heatmap.device)

        # 计算每个点到最大置信度点 (y, x) 的距离
        distance = (Y - y) ** 2 + (X - x) ** 2

        # 生成高斯分布
        gaussian_map = torch.exp(-distance / (2 * sigma ** 2))

        # 将其赋值给对应的 batch
        gaussian_heatmap[b] = gaussian_map

    return gaussian_heatmap


import matplotlib.pyplot as plt
import torch


def visualize_heatmap_and_gaussian(heatmap, gaussian_heatmap, batch_index=0):
    """
    可视化原始热图和生成的高斯分布热图，验证高斯分布的正确性。

    Parameters:
    - heatmap (Tensor): 输入的热图，形状为 [B, H, W]
    - gaussian_heatmap (Tensor): 生成的高斯分布热图，形状为 [B, H, W]
    - batch_index (int): 要查看的批次索引

    """
    # 获取原始热图和高斯热图
    original_map = heatmap[batch_index].cpu().numpy()
    gaussian_map = gaussian_heatmap[batch_index].cpu().numpy()

    # 设置图形大小
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 可视化原始热图
    axes[0].imshow(original_map, cmap='hot', interpolation='nearest')
    axes[0].set_title('Original Heatmap')
    axes[0].axis('off')

    # 可视化生成的高斯分布
    axes[1].imshow(gaussian_map, cmap='hot', interpolation='nearest')
    axes[1].set_title('Generated Gaussian Heatmap')
    axes[1].axis('off')

    # 显示图形
    plt.show()


# 示例：生成一个随机的热图，生成对应的高斯分布，并进行可视化
B, H, W = 1, 155, 155  # 批次大小为 1，图像大小为 64x64
heatmap = torch.rand(B, H, W)  # 生成一个随机热图

# 使用上面的函数生成高斯分布
gaussian_heatmap = generate_gaussian_heatmap(heatmap, sigma=1.0)

# 可视化
visualize_heatmap_and_gaussian(heatmap, gaussian_heatmap, batch_index=0)
