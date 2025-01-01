import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2


def softmax(x):
    """ Softmax function to convert distance map to probability map. """
    exp_x = np.exp(-x)  # 取负值，因为距离越小，置信度越高
    return exp_x / np.sum(exp_x)


def visualize_distance_map(satellite_img, distance_map, gt_point, alpha=0.7):
    """
    可视化 distance map 叠加在卫星图上，并标注 GT 点。

    Args:
        satellite_img (numpy.ndarray): 卫星图，shape 为 (H, W, 3)。
        distance_map (numpy.ndarray): 距离图，shape 为 (H, W)，值范围 0-4。
        gt_point (tuple): GT 点坐标，格式为 (x, y)。
        alpha (float): 叠加透明度，值范围 0-1。
    """
    # Step 1: Apply softmax to the distance map
    prob_map = (4-distance_map)/np.sum(distance_map)

    # Step 2: Check the probability map range
    print(f"Probability map min: {prob_map.min()}, max: {prob_map.max()}")  # 打印概率图的范围

    # Step 3: Create color map
    cmap = plt.cm.Reds  # 使用红色渐变色
    norm_prob_map = Normalize(vmin=0, vmax=1)(prob_map)  # Normalize to [0, 1]
    colored_prob_map = cmap(norm_prob_map)  # (H, W, 4)

    # Step 4: Check the values of the colored probability map
    print(f"Colored probability map sample (before conversion): {colored_prob_map[0, 0]}")  # 打印样本值

    # Convert RGBA to RGB
    colored_prob_map = (colored_prob_map[:, :, :3] * 255).astype(np.uint8)  # 转换到 RGB

    # Step 5: Check the RGB conversion
    print(f"Colored probability map sample (after conversion): {colored_prob_map[0, 0]}")  # 打印样本值

    # Step 6: Overlay probability map on satellite image
    satellite_img_rgb = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Use cv2.addWeighted for blending
    overlay = cv2.addWeighted(satellite_img_rgb, 1 - alpha, colored_prob_map, alpha, 0)

    # Step 7: Plot and mark GT point
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)  # Matplotlib 直接支持 RGB 图像
    plt.plot(gt_point[0], gt_point[1], marker='o', color='blue', markersize=5, label='GT Point')

    plt.axis('off')
    plt.title("Probability Map Visualization")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load satellite image
    sat_img_path = "/data/test/code/multi-local/satellite_40.7375031147_-73.9974854185.png"
    satellite_img = cv2.imread(sat_img_path)  # Default BGR format

    # Get the dimensions of the satellite image
    h, w, _ = satellite_img.shape

    # Generate distance map using Gaussian distribution
    mu, sigma = 2, 1  # Set mean and standard deviation
    distance_map = np.random.normal(mu, sigma, (h, w))  # Generate Gaussian noise


    # Clip values to be between 0 and 4
    distance_map = np.clip(distance_map, 0, 4)

    # Define GT point (replace with your actual GT point)
    gt_point = (100, 150)
    distance_map = distance_map.max() - distance_map

    # Visualize
    visualize_distance_map(satellite_img, distance_map, gt_point)
