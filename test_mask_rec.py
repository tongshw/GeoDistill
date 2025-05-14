import numpy as np
import matplotlib.pyplot as plt
import cv2  # 可选，用于可视化测试时读图

def mask_random_rectangle(image, mask_ratio=0.5):
    """
    随机在图像上mask一个面积为area_to_mask的矩形区域。

    参数：
    - image: numpy数组，输入图片，形状为(H, W)或(H, W, C)。
    - area_to_mask: int，欲遮挡的矩形区域面积（像素数）。

    返回：
    - masked_image: 被mask之后的图像。
    - mask: 与image相同尺寸的0-1矩阵，0表示被遮挡区域。
    """
    h, w = image.shape[:2]
    area_to_mask = int(h * w * mask_ratio)
    mask = np.ones((h, w), dtype=np.uint8)

    # 防止mask面积超过图像总面积
    area_to_mask = min(area_to_mask, h * w)

    for _ in range(100):  # 最多尝试100次寻找合适矩形
        aspect_ratio = np.random.uniform(0.3, 3.0)  # 宽高比可调节
        rect_h = int(np.sqrt(area_to_mask / aspect_ratio))
        rect_w = int(np.sqrt(area_to_mask * aspect_ratio))

        if rect_h < h and rect_w < w:
            top = np.random.randint(0, h - rect_h)
            left = np.random.randint(0, w - rect_w)
            mask[top:top + rect_h, left:left + rect_w] = 0
            break

    # 应用于原图
    if image.ndim == 3:
        masked_image = image.copy()
        masked_image[mask == 0] = 0
    else:
        masked_image = image * mask

    return masked_image, mask
# 示例测试（使用随机图）
if __name__ == "__main__":
    # 创建随机图像：彩色图 256x256
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # 或者用cv2读取图片测试：
    # test_img = cv2.imread("your_image.jpg")
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    area = 256 * 256 * 0.2  # 例如遮挡20%面积
    masked_img, mask = mask_random_rectangle(test_img, mask_ratio=0.5)

    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(test_img)
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Mask (0 = masked)")
    axs[2].imshow(masked_img)
    axs[2].set_title("Masked Image")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
