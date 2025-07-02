import copy
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.models as models

from model.efficientnet_pytorch import EfficientNet
from torchvision import transforms
from VGG import VGGUnet
import torchvision.transforms.functional as TF

from utils.util import grid_sample

from .dino import DINO
from .dpt import DPT

class LocalizationNet(nn.Module):
    def __init__(self, args, grid_size=8):
        super().__init__()

        self.levels = args.levels
        self.channels = args.channels

        input_dim = 3
        # self.sat_VGG = VGGUnet(self.levels, self.channels)
        # self.grd_VGG = VGGUnet(self.levels, self.channels) if args.p_siamese else None
        self.SatDPT = DPT()
        self.GrdDPT = DPT()

        feature_dim = 320
        self.rotation_range = 0
        self.grd_height = -2

    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        # shift_u = shift_u / np.power(2, 3 - level)
        # shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        # shift_u = shift_u / 512 * S
        # shift_v = shift_v / 512 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1))) ** 2 + (
                jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1))) ** 2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)),
                            jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = theta % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3 - level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(self.grd_height))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_feature, grd_confidence, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_feature.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_feature, uv)
        if grd_confidence is not None:
            grd_c_trans, _ = grid_sample(grd_confidence, uv)
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv

    def forward(self, sat_feat_list, pano_feat_list, meter_per_pixel, mask=None):
        B = sat_feat_list[0].shape[0]
        # shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        # shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        # grd1_proj, grd1_conf_proj, grd_uv = self.project_grd_to_map(
        #     pano1.permute(0, 3, 1, 2), pano1, None, shift_u, shift_v, 2, meter_per_pixel)
        # plt.figure(figsize=(10, 5))
        # plt.imshow(grd1_proj[1].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
        # plt.show()

        sat_feat_dict = self.SatDPT(sat_feat_list)

        pano_feat_dict = self.GrdDPT(pano_feat_list)

        g2s_feat_dict = {}
        g2s_conf_dict = {}
        mask_dict = {}
        pano_conf_dict = {}
        sat_conf_dict = {}
        # corr_maps = {}
        B = sat_feat_dict[0].shape[0]

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_feat_dict[0].device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_feat_dict[0].device)
        if mask is not None:
            mask = mask.cpu().numpy()

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            pano_feat = pano_feat_dict[level]
            pano_conf = torch.ones_like(pano_feat)

            B, c, h, w = pano_feat.shape
            A = sat_feat.shape[-1]
            crop_H = int(A * 0.4)
            crop_W = int(A * 0.4)

            if mask is not None:
                resized_mask_batch = []

                for i in range(B):  # 遍历 batch
                    resized_mask = cv2.resize(mask[i], (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_mask_batch.append(resized_mask)

                # 将结果转回 NumPy 数组，然后再转回 PyTorch 张量
                resized_mask_batch = np.stack(resized_mask_batch)  # 将所有图片拼接成一个数组
                mask_ = torch.from_numpy(resized_mask_batch)
                mask_ = mask_.to(sat_feat.device).permute(0, 3, 1, 2)
                mask_proj, _, grd_uv = self.project_grd_to_map(
                    mask_, pano_conf, None, shift_u, shift_v, level, meter_per_pixel)
                mask_ = TF.center_crop(mask_proj, [crop_H, crop_W])
                mask_dict[level] = mask_

            grd_feat_proj, grd_conf_proj, grd_uv = self.project_grd_to_map(
                pano_feat, pano_conf, None, shift_u, shift_v, level, meter_per_pixel)

            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_conf = TF.center_crop(grd_conf_proj, [crop_H, crop_W])

            g2s_feat_dict[level] = g2s_feat
            g2s_conf_dict[level] = g2s_conf
            sat_conf_dict[level] = g2s_conf

        return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict, pano_conf_dict, pano_feat_dict

    def calc_corr_for_train(self, sat_feat_dict, bev_feat_dict, mask_dict=None,batch_wise=False):
        corr_maps = {}

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            # sat_conf = sat_conf_dict[level]
            bev_feat = bev_feat_dict[level]
            bev_conf = torch.ones_like(bev_feat)[:, :1, :, :]
            if mask_dict is not None:
                mask = mask_dict[level]

                mask = mask[:, 0, :, :]
                # plt.figure(figsize=(4, 4))  # 设置图大小
                # plt.imshow(mask[0].cpu().detach().numpy(), cmap="viridis")  # 使用 viridis 颜色映射
                # plt.colorbar(label="Confidence")  # 添加颜色条
                # plt.title(f"bev mask ")
                # plt.axis("on")  # 关闭坐标轴
                # plt.show()
                mask = mask.unsqueeze(1)

                bev_conf = bev_conf * mask
                mask = mask.repeat(1, bev_feat.shape[1], 1, 1)
                bev_feat = bev_feat * mask
            B = bev_feat.shape[0]
            A = sat_feat.shape[2]

            if batch_wise:
                signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W] [8, 2048, 64, 64]
                kernel = bev_feat * bev_conf.pow(2)  # [8, 256, 25, 25]
                corr = F.conv2d(signal, kernel, groups=B)  # [8, 8, 40, 40], B=8

                # denominator
                denominator_sat = []
                sat_feat_pow = (sat_feat).pow(2)
                bev_conf_pow = bev_conf.pow(2)
                for i in range(0, B):
                    denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], bev_conf_pow), dim=0)
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]

                denom_grd = torch.linalg.norm((bev_feat * bev_conf).reshape(B, -1), dim=-1)  # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

                denominator = denominator_sat * denominator_grd

                denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)
            else:

                signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
                kernel = bev_feat * bev_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

                # denominator
                sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
                g2s_conf_pow = bev_conf.pow(2)
                denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
                denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

                denom_grd = torch.linalg.norm((bev_feat * bev_conf).reshape(B, -1), dim=-1)  # [B]
                shape = denominator_sat.shape
                denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

                denominator = denominator_sat * denominator_grd

            corr = 2 - 2 * corr / denominator  # [B, B, H, W]

            corr_maps[level] = corr

        return corr_maps

    def calc_corr_for_val(self, sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask_dict=None):
        level = self.levels[-1]

        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        bev_feat = bev_feat_dict[level]
        bev_conf = torch.ones_like(bev_feat)[:, :1, :, :]
        mask = None
        if mask_dict is not None:
            mask = mask_dict[level]

        B, C, crop_H, crop_W = bev_feat.shape
        A = sat_feat.shape[2]

        if mask is not None:
            mask = mask[:, 0, :, :]
            # plt.figure(figsize=(4, 4))  # 设置图大小
            # plt.imshow(mask[0].cpu().detach().numpy(), cmap="viridis")  # 使用 viridis 颜色映射
            # plt.colorbar(label="Confidence")  # 添加颜色条
            # plt.title(f"bev mask ")
            # plt.axis("on")  # 关闭坐标轴
            # plt.show()
            mask = mask.unsqueeze(1)

            bev_conf = bev_conf * mask
            mask = mask.repeat(1, bev_feat.shape[1], 1, 1)
            bev_feat = bev_feat * mask

        signal = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
        kernel = bev_feat * bev_conf.pow(2)
        corr = F.conv2d(signal, kernel, groups=B)[0]  # [B, H, W]

        # denominator
        sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
        g2s_conf_pow = bev_conf.pow(2)
        denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

        denom_grd = torch.linalg.norm((bev_feat * bev_conf).reshape(B, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

        denominator = denominator_sat * denominator_grd

        corr = corr / denominator

        return corr
