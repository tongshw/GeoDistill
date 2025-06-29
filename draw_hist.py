# import os
# import matplotlib.pyplot as plt
#
# directory = "/data/test/code/geodistill_vit/vis/error_analysis/cross-g2s-inversedataset-infer-s_20250613_115509/Seattle"
#
# # 获取最后一级目录名
# city_name = os.path.basename(directory)
#
# # 存储结果的字典
# file_counts = {}
#
# # 遍历目录
# for folder in os.listdir(directory):
#     folder_path = os.path.join(directory, folder)
#     if os.path.isdir(folder_path):
#         try:
#             number = int(folder.split('-')[0])
#             count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
#             file_counts[number] = count
#         except:
#             continue
#
# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.bar(list(file_counts.keys()), list(file_counts.values()))
# plt.xlabel('error meters')
# plt.ylabel('samples')
# plt.title(f'error distribution-{city_name}')
# plt.grid(True)
# plt.show()
import os
import matplotlib.pyplot as plt

parent_directory = "/data/test/code/geodistill_vit/vis/error_analysis/cross-g2s-inversedataset-infer-s_20250613_115509"

# 创建图形
plt.figure(figsize=(12, 7))

# 获取所有城市目录
cities = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

# 为每个城市选择不同的颜色和透明度
for i, city in enumerate(cities):
    city_path = os.path.join(parent_directory, city)
    file_counts = {}

    # 遍历每个城市的子目录
    for folder in os.listdir(city_path):
        folder_path = os.path.join(city_path, folder)
        if os.path.isdir(folder_path):
            try:
                number = int(folder.split('-')[0])
                count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                file_counts[number] = count
            except:
                continue

    # 绘制该城市的直方图
    plt.bar(list(file_counts.keys()),
            list(file_counts.values()),
            alpha=0.5,  # 设置透明度
            label=city)  # 添加图例标签

plt.xlabel('error meters')
plt.ylabel('samples')
plt.title('G2SWeakly Error Distribution Comparison')
plt.grid(True)
plt.legend()  # 显示图例
plt.show()
