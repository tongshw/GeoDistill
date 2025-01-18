import os

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_distributions(distance_t, distance_s, bins, output_dir):
    """
    可视化两个数组的累计统计、区间统计（以百分比显示），以及它们差值的区间统计（以百分比显示），并将图片保存到本地。
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()

    # print(f"Plots saved in {output_dir}")

# 示例调用
# 假设你有以下数据
pred_us_t = np.random.rand(100) * 10
pred_vs_t = np.random.rand(100) * 10
pred_us_s = np.random.rand(100) * 10
pred_vs_s = np.random.rand(100) * 10
gt_us = np.random.rand(100) * 10
gt_vs = np.random.rand(100) * 10

# 指定区间边界
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 保存路径
output_dir = "./output"
distance_t = np.sqrt((pred_us_t - gt_us) ** 2 + (pred_vs_t - gt_vs) ** 2)  # [N]
distance_s = np.sqrt((pred_us_s - gt_us) ** 2 + (pred_vs_s - gt_vs) ** 2)  # [N]
# 调用函数
visualize_distributions(distance_t, distance_s, bins, output_dir)
