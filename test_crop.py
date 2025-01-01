import numpy as np

# 假设原始图片是一个大小为180x360的numpy数组
img = np.random.rand(180, 360)  # 示例随机生成的图片

# 设定区域的起始位置
start_col = 90  # 从第90列开始提取（假设提取中间部分）

# 提取连续区域（180x180）
region = img[:, start_col:start_col+180]

# 创建子图1：h=180, w=360，其中区域部分复制，其他部分用0填充
sub_img1 = np.zeros((180, 360))  # 初始化为0的数组
sub_img1[:, start_col:start_col+180] = region  # 复制区域

# 创建子图2：其余部分复制到子图2
sub_img2 = np.zeros((180, 360))  # 初始化为0的数组
sub_img2[:, :start_col] = img[:, :start_col]  # 复制左侧部分
sub_img2[:, start_col+180:] = img[:, start_col+180:]  # 复制右侧部分

# 输出子图1和子图2
print("子图1:\n", sub_img1)
print("子图2:\n", sub_img2)
