# import matplotlib.pyplot as plt
#
#
# def plot_with_axis_padding(data, names, y_pad=0.01, x_pad=0.5):
#     """ 带坐标轴留白的折线图
#     Parameters:
#         y_pad : y轴上方留白比例（默认多留30%空间）
#         x_pad : x轴右侧留白比例（默认多留50%空间）
#     """
#     plt.figure(figsize=(8, 4))
#
#     # 绘制所有折线
#     for values, label in zip(data, names):
#         plt.plot(values, label=label, marker='o', linewidth=2)
#
#     # 自动计算坐标轴范围
#     all_values = [num for sublist in data for num in sublist]
#     y_max = max(all_values)
#     x_max = len(data[0]) - 1  # 假设所有数据长度相同
#
#     # 设置坐标轴范围
#     plt.ylim(top=y_max * (1 + y_pad))  # y轴顶部留白
#     plt.xlim(right=x_max + x_pad)  # x轴右侧留白
#
#     # 添加图表元素
#     plt.title('Val error in training at VIGOR same-area', fontsize=13)
#     plt.xlabel('epoch', fontsize=13)
#     plt.ylabel('val error', fontsize=13)
#     plt.grid(ls=':')
#
#     # 保持图例在右上角
#     plt.legend(loc='upper right', fontsize=17)
#
#     plt.tight_layout()
#     plt.show()
#
#
#
#
# # 使用示例（自动留白）
# data_cross = [[5.2604, 5.2504, 5.0614, 5.2078, 4.9375, 4.9561, 4.9936, 5.0104, 5.1231, 5.0721,
#          4.9674, 4.9274, 5.0125, 4.9841, 4.8999, 4.8982, 4.8845],
#         [5.2840, 5.2299, 5.2238, 5.1902, 5.1192, 5.0710, 5.0315, 5.0309, 5.0198, 4.9848,
#          4.9615, 4.9524, 4.9311, 4.9228, 4.9364, 4.9390, 4.9063]]
#
# data_same = [[4.4081, 4.2072, 4.2234, 4.2295, 4.1855, 4.1513, 4.1447, 4.1474, 4.1703, 4.0732,
#          4.1610, 4.0494, 4.1419, 4.2056, 4.1987, 4.0974, 4.1730],
#
#         [4.4977, 4.4552, 4.4020, 4.3608, 4.3285, 4.2776, 4.2344, 4.2015, 4.1612, 4.1147,
#          4.0969, 4.0822, 4.0697, 4.0555, 4.0672, 4.0554, 4.0382]]
# names = ["student", "teacher"]
# plot_with_axis_padding(data_same, names)

import matplotlib.pyplot as plt

def plot_with_axis_padding(data, names, y_pad=0.01, x_pad=0.5, ax=None):
    """ 带坐标轴留白的折线图
    Parameters:
        y_pad : y轴上方留白比例（默认多留30%空间）
        x_pad : x轴右侧留白比例（默认多留50%空间）
        ax : 传入的Axes对象，用于绘制在同一画布上
    """
    # 使用传入的ax（如果没有传入则自动创建一个新的图形）
    if ax is None:
        ax = plt.gca()

    # 绘制所有折线
    for values, label in zip(data, names):
        ax.plot(values, label=label, marker='o', linewidth=2)

    # 自动计算坐标轴范围
    all_values = [num for sublist in data for num in sublist]
    y_max = max(all_values)
    x_max = len(data[0]) - 1  # 假设所有数据长度相同

    # 设置坐标轴范围
    ax.set_ylim(top=y_max * (1 + y_pad))  # y轴顶部留白
    ax.set_xlim(right=x_max + x_pad)  # x轴右侧留白

    # 添加图表元素
    ax.set_title('Val error in training at VIGOR same-area', fontsize=13)
    ax.set_xlabel('epoch', fontsize=13)
    ax.set_ylabel('val error', fontsize=13)
    ax.grid(ls=':')

    # 保持图例在右上角
    ax.legend(loc='upper right', fontsize=17)


# 使用示例（自动留白）
data_cross = [[5.2604, 5.2504, 5.0614, 5.2078, 4.9375, 4.9561, 4.9936, 5.0104, 5.1231, 5.0721,
         4.9674, 4.9274, 5.0125, 4.9841, 4.8999, 4.8982, 4.8845],
        [5.2840, 5.2299, 5.2238, 5.1902, 5.1192, 5.0710, 5.0315, 5.0309, 5.0198, 4.9848,
         4.9615, 4.9524, 4.9311, 4.9228, 4.9364, 4.9390, 4.9063]]

data_same = [[4.4081, 4.2072, 4.2234, 4.2295, 4.1855, 4.1513, 4.1447, 4.1474, 4.1703, 4.0732,
         4.1610, 4.0494, 4.1419, 4.2056, 4.1987, 4.0974, 4.1730],

        [4.4977, 4.4552, 4.4020, 4.3608, 4.3285, 4.2776, 4.2344, 4.2015, 4.1612, 4.1147,
         4.0969, 4.0822, 4.0697, 4.0555, 4.0672, 4.0554, 4.0382]]
names = ["student", "teacher"]

# 创建一个包含2行1列子图的画布
fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 2行1列，画布大小为8x8

# 绘制第一张图
plot_with_axis_padding(data_same, names, ax=axs[0])

# 绘制第二张图
plot_with_axis_padding(data_cross, names, ax=axs[1])

# 自动调整子图间的距离

plt.tight_layout()

# 显示图形
plt.show()
