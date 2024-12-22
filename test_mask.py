import cv2
import numpy as np

def generate_visible_masks(pano, fov_size):
    """
    生成两个连续的 mask，分别对应两片区域，满足重叠条件。

    参数：
    - pano: 输入的全景图像 (H, W, C)
    - fov_size: 可视区域的角度大小 (范围为360的分区数)

    返回：
    - resized_pano1: 带有第一个 mask 应用的全景图
    - resized_pano2: 带有第二个 mask 应用的全景图
    """

    h, w, c = pano.shape  # 获取图像尺寸

    # 第一个区域的起点和终点
    start_angle1 = np.random.randint(0, fov_size)
    w_start1 = int(np.round(w / 360 * start_angle1))
    w_end1 = int(np.round(w / 360 * (start_angle1 + 360 - fov_size)))

    if start_angle1 > 90:
        if (start_angle1 + 90 * 2) < 360:
            valid_range = list(range(0, start_angle1-90)) + list(range(start_angle1 + 90, 361 - 90))
        else:
            valid_range = list(range(0, start_angle1-90))
    else:
        valid_range = list(range(start_angle1 + 90, 360 - 90))

    start_angle2 = np.random.choice(valid_range)

    w_start2 = int(np.round(w / 360 * start_angle2))
    w_end2 = int(np.round(w / 360 * (start_angle2 + 360 - fov_size)))

    # 创建两个 mask，并应用到原图像上
    mask1 = np.zeros_like(pano)
    mask2 = np.zeros_like(pano)
    pano1 = pano.copy()
    pano2 = pano.copy()
    ones1 = np.ones_like(pano)
    ones2 = np.ones_like(pano)



    pano1[:, w_start1:w_end1, :] = mask1[:, w_start1:w_end1, :]
    pano2[:, w_start2:w_end2, :] = mask2[:, w_start2:w_end2, :]
    ones1[:, w_start1:w_end1, :] = mask1[:, w_start1:w_end1, :]
    ones2[:, w_start2:w_end2, :] = mask2[:, w_start2:w_end2, :]

    return pano1, pano2, ones1, ones2

# 示例使用
if __name__ == "__main__":
    # 生成示例全景图像
    pano = cv2.imread("/data/test/code/multi-local/1.jpg", 1)[:, :, ::-1]
    pano = cv2.resize(pano, (640, 320))

    fov_size = 270  # 可视区域大小
    resized_pano1, resized_pano2, ones1, ones2 = generate_visible_masks(pano, fov_size)

    # 检查区域的应用效果
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(40, 10))
    # ax[0].imshow(pano)
    # ax[0].set_title("Original Pano")
    ax[0].imshow(resized_pano1)
    ax[0].set_title("Pano with Mask 1")
    ax[1].imshow(resized_pano2)
    ax[1].set_title("Pano with Mask 2")
    ax[2].imshow(ones1)
    ax[2].set_title("Mask 1")
    ax[3].imshow(ones2)
    ax[3].set_title("Mask 2")
    plt.show()
