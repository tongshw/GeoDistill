import matplotlib.pyplot as plt
import torch
import math

def gaussian_label(grid_size, grid_x, grid_y, sigma=1.0):
    """
    生成一个高斯分布标签，用于可视化。
    grid_size: 网格的大小 (grid_size x grid_size)
    grid_x: 高斯分布中心的x坐标 (网格索引)
    grid_y: 高斯分布中心的y坐标 (网格索引)
    sigma: 高斯分布的标准差
    """
    grid_indices = torch.zeros((grid_size, grid_size))
    for y in range(grid_size):
        for x in range(grid_size):
            distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
            grid_indices[y, x] = math.exp(-distance / (2 * sigma ** 2))
    grid_indices /= grid_indices.sum()  # 归一化，使概率和为1
    return grid_indices

def visualize_gaussian_label(grid_size, grid_x, grid_y, sigma=1.0):
    """
    可视化高斯分布标签。
    """
    label = gaussian_label(grid_size, grid_x, grid_y, sigma)
    plt.figure(figsize=(6, 6))
    plt.imshow(label.numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Gaussian Label (grid_size={grid_size}, center=({grid_x},{grid_y}), sigma={sigma})')
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()

# 参数配置
grid_size = 8  # 网格大小，例如 16x16
grid_x, grid_y = 2, 5  # 高斯分布的中心位置 (在网格中的索引)
sigma = 0.8  # 高斯分布的标准差

# 可视化
visualize_gaussian_label(grid_size, grid_x, grid_y, sigma)
