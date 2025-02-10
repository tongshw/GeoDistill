import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import gridspec


def visualize_attention_map(attention_map_pano, attention_map_sat, pano, sat, gt_point=None, pred_point=None):
    # Squeeze batch dimension and ensure 2D input
    attention_map_pano = np.squeeze(attention_map_pano)
    attention_map_sat = np.squeeze(attention_map_sat)
    # Resize with interpolation
    attention_map_resized_pano = cv2.resize(attention_map_pano,
                                       (pano.shape[1], pano.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

    attention_map_resized_sat = cv2.resize(attention_map_sat,
                                            (sat.shape[1], sat.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1]
    attention_map_resized_pano = (attention_map_resized_pano - attention_map_resized_pano.min()) / \
                            (attention_map_resized_pano.max() - attention_map_resized_pano.min() + 1e-8)

    attention_map_resized_sat = (attention_map_resized_sat - attention_map_resized_sat.min()) / \
                            (attention_map_resized_sat.max() - attention_map_resized_sat.min() + 1e-8)

    # Convert image to RGB if needed
    if pano.shape[-1] != 3:
        pano = np.transpose(pano, (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 如果原图是BGR格式

    # Generate heatmap
    heatmap_pano = cv2.applyColorMap(np.uint8(255 * attention_map_resized_pano), cv2.COLORMAP_JET)
    heatmap_pano = cv2.cvtColor(heatmap_pano, cv2.COLOR_BGR2RGB)
    heatmap_pano = np.float32(heatmap_pano) / 255

    heatmap_sat = cv2.applyColorMap(np.uint8(255 * attention_map_resized_sat), cv2.COLORMAP_JET)
    heatmap_sat = cv2.cvtColor(heatmap_sat, cv2.COLOR_BGR2RGB)
    heatmap_sat = np.float32(heatmap_sat) / 255

    # Normalize image
    pano = np.float32(pano) / 255
    sat = np.float32(sat) / 255

    # Blend using OpenCV's weighted sum
    overlay_pano = cv2.addWeighted(pano, 0.4, heatmap_pano, 0.6, 0)
    overlay_sat = cv2.addWeighted(sat, 0.4, heatmap_sat, 0.6, 0)

    fig = plt.figure(figsize=(16, 16))  # 根据需要调整画布大小

    # 使用 gridspec 定义布局，改为两行两列
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 640 / 320], height_ratios=[1, 1])  # 设置宽高比例

    # 第一个子图：卫星图像和概率图叠加
    ax1 = fig.add_subplot(gs[0, 0])  # 指定第一个子图位置
    ax1.imshow(sat)  # Matplotlib 直接支持 RGB 图像
    if gt_point is not None:
        ax1.plot(gt_point[0].cpu().numpy(), gt_point[1].cpu().numpy(), marker='o', color='blue', markersize=8,
                 label='GT')
        ax1.legend(loc='upper right')
    if pred_point is not None:
        ax1.plot(pred_point[0].cpu().numpy(), pred_point[1].cpu().numpy(), marker='x', color='green', markersize=8,
                 label='pred')
        ax1.legend(loc='upper right')
    ax1.axis('on')
    ax1.set_title(f"Satellite Image with Probability Map")

    # 第二个子图：BEV图像（彩色 RGB）
    ax2 = fig.add_subplot(gs[0, 1])  # 指定第二个子图位置
    _, h, w = pano.shape
    if h == w:
        pano = pano.permute(1, 2, 0)
    # bev_img_rgb = bev.detach().cpu().numpy().astype(np.uint8)  # BEV 转换为 RGB
    ax2.imshow(pano)
    ax2.axis('on')
    ax2.set_title("BEV Image")

    # 第三个子图（可以自定义）
    ax3 = fig.add_subplot(gs[1, 0])  # 指定第三个子图位置
    ax3.imshow(overlay_sat)  # Matplotlib 直接支持 RGB 图像
    if gt_point is not None:
        ax3.plot(gt_point[0].cpu().numpy(), gt_point[1].cpu().numpy(), marker='o', color='blue', markersize=8,
                 label='GT')
        ax3.legend(loc='upper right')
    if pred_point is not None:
        ax3.plot(pred_point[0].cpu().numpy(), pred_point[1].cpu().numpy(), marker='x', color='green', markersize=8,
                 label='pred')
        ax3.legend(loc='upper right')
    ax3.axis('on')
    ax3.set_title(f"Satellite Image with Probability Map")

    # 第四个子图（可以自定义）
    ax4 = fig.add_subplot(gs[1, 1])  # 指定第四个子图位置
    # bev_img_rgb = bev.detach().cpu().numpy().astype(np.uint8)  # BEV 转换为 RGB
    ax4.imshow(overlay_pano)
    ax4.axis('on')
    ax4.set_title("BEV Image")


    # 调整布局
    plt.tight_layout()
    plt.show()

    # # Plotting
    # plt.figure(figsize=(5, 5))
    # plt.subplot(2, 1, 1), plt.imshow(pano), plt.axis('off')
    # plt.subplot(2, 1, 2), plt.imshow(overlay_pano), plt.axis('off')
    # plt.tight_layout()
    # plt.show()






# Example usage
# attention_map has shape (1, H/2, W/2)
# image has shape (3, H, W)
# attention_map = np.random.rand(1, 256, 256)  # Example random attention map
# image = np.random.rand(3, 512, 512)  # Example random image
#
# visualize_attention_map(attention_map, image)
