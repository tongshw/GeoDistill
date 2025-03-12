import matplotlib.pyplot as plt

def plot_line_chart(y_data):
    # 生成x轴坐标（30, 60, 90,..., 270）
    x = list(range(30, 271, 30))

    # 检查输入数据长度是否匹配
    if len(y_data) != len(x):
        raise ValueError(f"输入数组长度应为 {len(x)}，当前为 {len(y_data)}")

    # 创建画布和坐标轴
    plt.figure(figsize=(12, 4))  # 调整figsize的高度，降低纵横比

    # 绘制折线图（蓝色实线，带数据点标记）
    plt.plot(x, y_data, 'b-o', linewidth=2, markersize=8)

    # 设置坐标轴属性
    xtick_labels = [f"{num}°" for num in x]
    plt.xticks(x, xtick_labels, fontsize=21)  # 调大x轴刻度字体
    plt.yticks(fontsize=21)  # 调大y轴刻度字体

    plt.title("VIGOR Cross Area", fontsize=25)  # 可选：调大标题字体
    plt.xlabel("FoV", fontsize=23)
    plt.ylabel("Mean error(m)", fontsize=23)

    # 设置 y 轴范围，使得 y 轴“压缩”
    plt.ylim(min(y_data) - 0.1, max(y_data) + 0.1)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()

# 示例数据
example_data = [5.17, 4.93, 4.71, 4.64, 4.62, 4.59, 4.66, 4.68, 4.80]
plot_line_chart(example_data)
