import matplotlib.pyplot as plt


def plot_with_axis_padding(data, names, y_pad=0.3, x_pad=0.5):
    """ 带坐标轴留白的折线图
    Parameters:
        y_pad : y轴上方留白比例（默认多留30%空间）
        x_pad : x轴右侧留白比例（默认多留50%空间）
    """
    plt.figure(figsize=(8, 5))

    # 绘制所有折线
    for values, label in zip(data, names):
        plt.plot(values, label=label, marker='o', linewidth=2)

    # 自动计算坐标轴范围
    all_values = [num for sublist in data for num in sublist]
    y_max = max(all_values)
    x_max = len(data[0]) - 1  # 假设所有数据长度相同

    # 设置坐标轴范围
    plt.ylim(top=y_max * (1 + y_pad))  # y轴顶部留白
    plt.xlim(right=x_max + x_pad)  # x轴右侧留白

    # 添加图表元素
    plt.title('Line Chart with Axis Padding')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(ls=':')

    # 保持图例在右上角
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# 使用示例（自动留白）
data = [[1.2, 1.3, 3, 4], [1.1, 2, 1.5, 3]]
names = ["teacher", "student"]
plot_with_axis_padding(data, names)
