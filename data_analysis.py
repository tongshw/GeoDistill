import numpy as np
from matplotlib import pyplot as plt


def calculate_mean_for_range(start, end, A, B):
    # 找到 A 中在区间 [start, end) 内的元素的索引
    indices = [i for i, value in enumerate(A) if start <= value < end]

    # 计算 A 和 B 中对应元素的均值
    A_mean = np.mean([A[i] for i in indices]) if indices else 0
    B_mean = np.mean([B[i] for i in indices]) if indices else 0

    return A_mean, B_mean


distance_t = np.load('distance_t_cross.npy')
distance_s = np.load('distance_s_cross.npy')

for start in range(0, 20):
    end = start + 1
    A_mean, B_mean = calculate_mean_for_range(start, end, distance_t, distance_s)
    print(f"A中误差在{start}~{end}区间的元素均值: {A_mean}")
    print(f"B中对应元素的均值: {B_mean}")
    print("------")

indices_0_to_1 = [i for i, value in enumerate(distance_t) if 0 <= value < 1]

# 统计比这些符合条件的distance_t值小或大的distance_s值的个数
less_than_count = 0
greater_than_count = 0
total_count = 0

for idx in indices_0_to_1:
    t_value = distance_t[idx]

    # 对应的distance_s中的值
    s_value = distance_s[idx]

    # 统计比t_value小和大的元素
    less_than_count += sum(1 for value in distance_s if value < t_value)
    greater_than_count += sum(1 for value in distance_s if value > t_value)
    total_count += 1  # 记录符合条件的distance_t的元素个数

# 计算百分比
total_distance_s_count = len(distance_s)
less_than_percent = (less_than_count / (total_distance_s_count * total_count)) * 100
greater_than_percent = (greater_than_count / (total_distance_s_count * total_count)) * 100

# 输出总体结果
print(f"distance_t中值在0~1区间的元素，distance_s中比它们小的百分比: {less_than_percent:.2f}%")
print(f"distance_t中值在0~1区间的元素，distance_s中比它们大的百分比: {greater_than_percent:.2f}%")
error_reduction = distance_t - distance_s  # 正值表示误差减小

# 统计误差减小的点
num_improved = np.sum(error_reduction > 0)  # 有多少点的误差减小了
num_total = len(distance_t)

num_worse = np.sum(error_reduction < 0)

# 误差减小了的百分比
improvement_percentage = num_improved / num_total * 100
worse_percentage = num_worse / num_total * 100

# 计算误差减小的平均量
average_error_reduction = np.mean(error_reduction[error_reduction > 0])
average_err_increase = np.mean(error_reduction[error_reduction < 0])

# 计算误差减小的最大值和最小值
max_error_reduction = np.max(error_reduction[error_reduction > 0])
min_error_reduction = np.min(error_reduction[error_reduction > 0])
max_err_increase = np.min(error_reduction[error_reduction < 0])

print(f"总点数: {num_total}")
print(f"误差减小的点数: {num_improved}")
print(f"worse: {num_worse}")
print(f"误差减小的百分比: {improvement_percentage:.2f}%")
print(f"worse percentage: {worse_percentage:.2f}%")
print(f"误差减小的平均量: {average_error_reduction:.4f}")
print(f"worse average error: {average_err_increase:.4f}")
print(f"误差减小的最大值: {max_error_reduction:.8f}")
print(f"worse max error: {max_err_increase:.8f}")
print(f"误差减小的最小值: {min_error_reduction:.8f}")

error_reduction = distance_t - distance_s  # 正值表示误差减小，负值表示误差增大

error_reduction = error_reduction[error_reduction != 0]

# 设置区间范围（bins），只在-10到10范围内，区间宽度为0.5
bins = np.arange(-40, 41, 1)

# 计算每个区间的点数
hist, bin_edges = np.histogram(error_reduction, bins=bins)

# 计算柱状条的中心位置
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 设置截断值
max_height = 1000
hist_clipped = np.clip(hist, 0, max_height)  # 将柱状条高度限制到最大值

# 绘制柱状图
plt.figure(figsize=(10, 6))

# 绘制左侧（Δε < 0）的柱状图，使用粉色
plt.bar(bin_centers[bin_centers < 0], hist_clipped[bin_centers < 0], width=1,
        color='magenta', label=r"$\Delta \epsilon < 0$ (teacher is better)", align='center')

# 绘制右侧（Δε > 0）的柱状图，使用橙色
plt.bar(bin_centers[bin_centers >= 0], hist_clipped[bin_centers >= 0], width=1,
        color='orange', label=r"$\Delta \epsilon > 0$ (student is better)", align='center')

# 在柱状图顶部标注超出截断值的真实数值
for i, h in enumerate(hist):
    if h > max_height:  # 如果原始值超过截断值
        plt.text(bin_centers[i], max_height + 5, f"{h}", ha='center', va='bottom', fontsize=8, color='black')

# 设置图例和标题
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)  # 中间的竖线
plt.legend(fontsize=12)
plt.title(r"Error Reduction $\Delta \epsilon$ Distribution (-10 to 10), step0.5", fontsize=14)
plt.xlabel(r"$\Delta \epsilon$ (Error Reduction)", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)

# 设置网格线和坐标轴范围
plt.grid(alpha=1)
plt.xlim(-40, 40)
plt.ylim(0, max_height + 30)  # 给y轴留点空隙，用于显示标注

# 展示图表
plt.show()