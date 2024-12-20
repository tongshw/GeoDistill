import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

class TextColors:
    RED = '31'
    GREEN = '32'
    YELLOW = '33'
    BLUE = '34'
    MAGENTA = '35'
    CYAN = '36'
    WHITE = '37'

def print_colored(text, color=TextColors.RED):
    print(f"\033[{color}m{text}\033[0m")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualization(bev, sat, sat_delta, ori_angle):
    batch_size = bev.shape[0]  # 批量大小为 2
    bev = bev.cpu().numpy()  # 转换为 NumPy
    sat = sat.cpu().numpy()  # 转换为 NumPy
    sat_delta = sat_delta.cpu().numpy()  # 假设形状为 (batch_size, 2)
    ori_angle = ori_angle.cpu().numpy()  # 假设形状为 (batch_size,)

    # 创建可视化
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))  # 每个样本两列

    # 遍历每个 batch 的数据
    for i in range(batch_size):
        print(sat_delta[i])
        # 处理 BEV 图像 (C, H, W) -> (H, W, C)
        bev_sample = bev[i].transpose(1, 2, 0)
        bev_sample = (bev_sample - bev_sample.min()) / (bev_sample.max() - bev_sample.min())  # 归一化

        # 处理 SAT 图像 (C, H, W) -> (H, W, C)
        sat_sample = sat[i].transpose(1, 2, 0)
        sat_sample = (sat_sample - sat_sample.min()) / (sat_sample.max() - sat_sample.min())  # 归一化

        # 绘制 BEV 图像
        axes[i, 0].imshow(bev_sample)
        axes[i, 0].set_title(f"BEV (ori_angle: {ori_angle[i]:.2f})", fontsize=12)
        axes[i, 0].axis("off")

        # 绘制 SAT 图像
        axes[i, 1].imshow(sat_sample)
        axes[i, 1].scatter(
            sat_delta[i][0],  # sat_delta_x 偏移
            sat_delta[i][1],  # sat_delta_y 偏移
            color="red",
            label="sat_delta"
        )
        axes[i, 1].set_title(f"SAT (ori_angle: {ori_angle[i]:.2f})", fontsize=12)
        axes[i, 1].legend()
        axes[i, 1].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()



def grid_sample(image, optical, jac=None):
    # values in optical within range of [0, H], and [0, W]
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(N, 1, H, W)
    iy = optical[..., 1].view(N, 1, H, W)

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x
        iy_nw = torch.floor(iy)  # north-west  upper-left-y
        ix_ne = ix_nw + 1        # north-east  upper-right-x
        iy_ne = iy_nw            # north-east  upper-right-y
        ix_sw = ix_nw            # south-west  lower-left-x
        iy_sw = iy_nw + 1        # south-west  lower-left-y
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0

    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:

        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # assert jac.shape[1:] == [N, H, W, 2]
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        if torch.any(torch.isnan(jac)) or torch.any(torch.isnan(dout_dpxy)):
            print('Nan occurs')

        return out_val, jac_new1 #jac_new1 #jac_new.permute(4, 0, 1, 2, 3)
    else:
        return out_val, None
