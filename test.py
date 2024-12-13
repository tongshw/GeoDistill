import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from geometry import get_BEV_projection, get_BEV_tensor

# 加载本地图片
image_paths = ["/data/test/code/multi-local/1.jpg", "/data/test/code/multi-local/1.jpg"]  # 替换为你的图片路径
images = []

# 预处理：将图片调整为相同大小并转为张量
def preprocess_image(image_path, target_size=(256, 256)):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

for path in image_paths:
    images.append(preprocess_image(path))

# 将所有图片堆叠成一个 batch
tensor_images = torch.stack(images)  # 形状: (batch_size, channels, height, width)

# 获取张量的形状
batch_size, channels, H, W = tensor_images.shape

# 计算中间区域的索引
w_start, w_end = W // 360 * 200, W // 360 * 360

# 创建一个与原图片相同大小的零张量
result = torch.zeros_like(tensor_images)

# 将中间区域的内容复制到结果张量中
result[:, :, :, w_start:w_end] = tensor_images[:, :, :, w_start:w_end]

# 可视化函数
def visualize_results(original, modified, idx):
    """可视化原始图片和修改后的图片"""
    original = transforms.ToPILImage()(original)
    # modified = transforms.ToPILImage()(modified)
    src = modified.permute(1, 2, 0).numpy()
    src *= 255
    src = src.astype(np.uint8)
    out = get_BEV_projection(src, 500, 500, Fov=85 * 2, dty=0, dx=0, dy=-10)
    BEV = get_BEV_tensor(src, 500, 500, Fov=85 * 2, dty=0, dx=0, dy=0, out=out).cpu().numpy().astype(np.uint8)
    BEV = cv2.resize(BEV, (430, 430))
    plt.imshow(BEV)
    plt.axis('on')
    plt.show()

# 可视化每张图片
for i in range(batch_size):
    visualize_results(tensor_images[i], result[i], i)
