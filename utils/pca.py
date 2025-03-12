import skimage.io as io
import os.path
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from matplotlib.cm import get_cmap
import torch
#pca方式完成特征图的可视化
#pcl_features_to_RGB([feature_map], 0, "result_visualize/")
def pcl_features_to_RGB(grd_feat_input, loop=0, save_dir="result_visualize/"):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator



    grd_feat = grd_feat_input.data.cpu().numpy()  # [B, C, H, W]

    B, C, H, W = grd_feat.shape


    pca_grd = PCA(n_components=3)
    pca_grd.fit(reshape_normalize(grd_feat))

    grd_feat = grd_feat_input.data.cpu().numpy()  # [B, C, H, W]

    B, C, H, W = grd_feat.shape
    grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat))) + 1) / 2).reshape(B, H, W, 3)

    for idx in range(B):
        if not os.path.exists(os.path.join(save_dir)):
            os.makedirs(os.path.join(save_dir))
        img_array = (grd_feat_new[idx] * 255).astype(np.uint8)
        grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
        grd = grd.resize((W,H))
        grd.save(save_dir + ('feat_' + str(loop * B + idx) + '.jpg'))
        plt.figure(figsize=(5, 5))
        plt.imshow(img_array)
        plt.axis("off")
        plt.title(f"Feature Map {loop * B + idx}")
        plt.show()

    return grd_feat_new

# a = torch.randn(2, 3, 224, 224)
# pcl_features_to_RGB(a, 0, "result_visualize/")