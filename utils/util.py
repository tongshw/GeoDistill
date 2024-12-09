import random

import numpy as np
import torch
from matplotlib import pyplot as plt


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