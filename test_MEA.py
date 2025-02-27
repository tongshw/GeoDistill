import cv2
import torch
from matplotlib import pyplot as plt

import torch


def patchify(imgs):
    """
    处理宽为高度两倍的图像的分块函数
    imgs: (N, 3, H, W) 且 W = 2H
    x: (N, L, patch_size**2 *3)
    """
    p = 16
    H, W = imgs.shape[2], imgs.shape[3]
    assert H % p == 0 and W % p == 0, "H and W must be divisible by patch size"
    assert W == 2 * H, "Width must be exactly twice the height"

    h = H // p
    w = W // p  # 当 W=2H 时，w = 2h
    x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
    return x


def unpatchify(x):
    """
    处理宽为高度两倍图像的还原函数
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W) 且 W = 2H
    """
    p = 16
    L = x.shape[1]
    # 根据 W=2H 的约束计算 h 和 w
    h = int((L / 2) ** 0.5)  # 由 L = h * w = h*(2h) = 2h² 推导
    w = 2 * h
    assert h * w == L, "Invalid number of patches for W=2H ratio"

    x = x.reshape(x.shape[0], h, w, p, p, 3)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], 3, h * p, w * p)
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


import numpy as np


def generate_MAE_mask(image, mask_ratio=0.75, patch_size=16):
    """
    生成与输入图片形状相同的mask，随机将部分16x16区域置0（覆盖75%面积），其余置1。

    参数：
    image: numpy数组，输入图片，形状为(H, W)或(H, W, C)。
    mask_ratio: float，需覆盖的面积比例，默认为0.75。
    patch_size: int，每个区域的大小，默认为16。

    返回：
    mask: numpy数组，与image形状相同的0-1矩阵。
    """
    h, w = image.shape[:2]
    target_area = mask_ratio * h * w  # 计算需要覆盖的总面积

    # 生成所有可能的块
    blocks = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            y_end = min(i + patch_size, h)
            x_end = min(j + patch_size, w)
            block_area = (y_end - i) * (x_end - j)
            blocks.append((i, j, y_end, x_end, block_area))

    # 随机打乱块顺序
    np.random.shuffle(blocks)

    # 选择块直到覆盖足够面积
    masked_area = 0
    selected_blocks = []
    for block in blocks:
        if masked_area >= target_area:
            break
        selected_blocks.append(block)
        masked_area += block[4]

    # 创建全1的mask
    mask = np.ones((h, w), dtype=np.uint8)

    # 将选中的块置0
    for y_start, x_start, y_end, x_end, _ in selected_blocks:
        mask[y_start:y_end, x_start:x_end] = 0

    # 扩展mask维度以匹配输入图片的形状（若为多通道）
    # if len(image.shape) == 3:
    #     mask = mask[:, :, np.newaxis].repeat(image.shape[2], axis=2)

    return mask


import torch


def generate_batch_mask(images, mask_ratio=0.75, patch_size=16):
    """
    生成与输入batch形状相同的mask，随机将部分16x16区域置0（覆盖75%面积），其余置1

    参数：
    images: torch.Tensor，输入图片batch，形状为(B, C, H, W)
    mask_ratio: float，需覆盖的面积比例，默认为0.75
    patch_size: int，每个区域的大小，默认为16

    返回：
    mask: torch.Tensor，形状为(B, 1, H, W)的0-1矩阵
    """
    B, C, H, W = images.shape
    device = images.device
    total_pixels = H * W
    target_area = mask_ratio * total_pixels

    # 生成所有块信息 ----------------------------------------------------------
    blocks = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            y_end = min(i + patch_size, H)
            x_end = min(j + patch_size, W)
            block_area = (y_end - i) * (x_end - j)
            blocks.append((i, j, y_end, x_end, block_area))
    num_blocks = len(blocks)

    # 批量随机选择逻辑 --------------------------------------------------------
    # 生成随机排序 (B, num_blocks)
    rand_matrix = torch.rand(B, num_blocks, device=device)
    sorted_indices = torch.argsort(rand_matrix, dim=1)  # 每个样本的块随机顺序

    # 计算累积面积
    block_areas = torch.tensor([b[4] for b in blocks], device=device)
    sorted_areas = block_areas[sorted_indices]  # (B, num_blocks)
    cum_areas = torch.cumsum(sorted_areas, dim=1)  # (B, num_blocks)

    # 确定每个样本需要选中的块数
    mask_needed = cum_areas >= target_area
    selected_num = torch.argmax(mask_needed.int(), dim=1)  # (B,)
    selected_num = torch.where(mask_needed.any(dim=1),
                               selected_num + 1,  # 包含触发点的块
                               torch.tensor(num_blocks, device=device))

    # 生成选择矩阵 (B, num_blocks)
    batch_indices = torch.arange(B, device=device)[:, None]
    selection_mask = torch.zeros_like(mask_needed)
    selection_mask[batch_indices, sorted_indices] = torch.arange(num_blocks, device=device) < selected_num[:, None]

    # 构建最终mask ----------------------------------------------------------
    mask = torch.ones(B, 1, H, W, device=device)
    for block_idx, (i, j, y_end, x_end, _) in enumerate(blocks):
        # 获取选择当前块的样本
        selected_samples = selection_mask[:, block_idx].nonzero(as_tuple=True)[0]
        if len(selected_samples) > 0:
            mask[selected_samples, :, i:y_end, j:x_end] = 0

    return mask
#
# sat_img_path = "/data/test/code/multi-local/1.jpg"
# satellite_img = cv2.imread(sat_img_path)  # Default BGR format
# plt.figure(figsize=(10, 5))
# plt.imshow(satellite_img)
# plt.show()
# mask = generate_MAE_mask(satellite_img)
# # satellite_img = torch.from_numpy(satellite_img)
# # satellite_img = satellite_img.unsqueeze(0).permute(0, 3, 1, 2)
# # satellite_img = patchify(satellite_img)
#
# # x_masked, mask, ids_restore = random_masking(satellite_img, 0.75)
# #
# # satellite_img = unpatchify(x_masked)
#
# mask = generate_MAE_mask(satellite_img)
#
# # 应用mask到图片
# masked_image = satellite_img * mask
# plt.figure(figsize=(10, 5))
# plt.imshow(masked_image)
# plt.show()