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


class LocalizationNet(nn.Module):
    def __init__(self, args, grid_size=8):
        super().__init__()

        self.levels = args.levels
        self.channels = args.channels

        # 保持原有的EfficientNet特征提取器
        input_dim = 3
        self.sat_VGG = VGGUnet(self.levels, self.channels)
        self.grd_VGG = VGGUnet(self.levels, self.channels) if args.p_siamese else None

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

    def forward_2grd(self, sat_img, pano1, ones1, pano2, ones2, meter_per_pixel):

        sat_img = 2 * (sat_img / 255.0) - 1.0
        pano1_img = 2 * (pano1 / 255.0) - 1.0
        pano2_img = 2 * (pano2 / 255.0) - 1.0
        # sat_img = sat_img / 255.0
        # pano1_img = pano1 / 255.0
        # pano2_img = pano2 / 255.0

        sat_img = sat_img.contiguous()
        pano1_img = pano1_img.contiguous()
        pano2_img = pano2_img.contiguous()

        pano1_img = pano1_img.permute(0, 3, 1, 2)
        pano2_img = pano2_img.permute(0, 3, 1, 2)
        pano1_zero = torch.sum(pano1_img == 0).item()

        sat_feat_dict, sat_conf_dict = self.sat_VGG(sat_img)
        pano1_feat_dict, pano1_conf_dict = self.grd_VGG(pano1_img)
        pano2_feat_dict, pano2_conf_dict = self.grd_VGG(pano2_img)
        feat1_zero = torch.sum(pano1_feat_dict[0] == 0).item()
        conf1_zero = torch.sum(pano1_conf_dict[0] == 0).item()

        g2s1_feat_dict = {}
        g2s1_conf_dict = {}
        g2s2_feat_dict = {}
        g2s2_conf_dict = {}
        mask1_dict = {}
        mask2_dict = {}
        # corr_maps = {}
        B = sat_conf_dict[0].shape[0]

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        mask1 = None
        mask2 = None
        ones1 = ones1.cpu().numpy()
        ones2 = ones2.cpu().numpy()

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            pano1_feat = pano1_feat_dict[level]
            pano1_conf = pano1_conf_dict[level]
            pano2_feat = pano2_feat_dict[level]
            pano2_conf = pano2_conf_dict[level]

            B, c, h, w = pano2_feat.shape

            resized_batch1 = []
            resized_batch2 = []

            for i in range(B):  # 遍历 batch
                resized_image1 = cv2.resize(ones1[i], (w, h), interpolation=cv2.INTER_LINEAR)
                resized_image2 = cv2.resize(ones2[i], (w, h), interpolation=cv2.INTER_LINEAR)
                resized_batch1.append(resized_image1)
                resized_batch2.append(resized_image2)

            # 将结果转回 NumPy 数组，然后再转回 PyTorch 张量
            resized_batch1 = np.stack(resized_batch1)  # 将所有图片拼接成一个数组
            mask1 = torch.from_numpy(resized_batch1)
            mask1 = mask1.to(sat_feat.device).permute(0, 3, 1, 2)

            resized_batch2 = np.stack(resized_batch2)  # 将所有图片拼接成一个数组
            mask2 = torch.from_numpy(resized_batch2)
            mask2 = mask2.to(sat_feat.device).permute(0, 3, 1, 2)

            # ones1 = cv2.resize(ones1, (w, h))
            # ones2 = cv2.resize(ones2, (w, h))

            grd1_feat_proj, grd1_conf_proj, grd_uv = self.project_grd_to_map(
                pano1_feat, pano1_conf, None, shift_u, shift_v, level, meter_per_pixel)

            grd2_feat_proj, grd2_conf_proj, grd_uv = self.project_grd_to_map(
                pano2_feat, pano2_conf, None, shift_u, shift_v, level, meter_per_pixel)

            mask1_proj, _, grd_uv = self.project_grd_to_map(
                mask1, pano1_conf, None, shift_u, shift_v, level, meter_per_pixel)
            mask2_proj, _, grd_uv = self.project_grd_to_map(
                mask2, pano2_conf, None, shift_u, shift_v, level, meter_per_pixel)

            A = sat_feat.shape[-1]
            crop_H = int(A * 0.4)
            crop_W = int(A * 0.4)
            g2s1_feat = TF.center_crop(grd1_feat_proj, [crop_H, crop_W])
            g2s1_conf = TF.center_crop(grd1_conf_proj, [crop_H, crop_W])
            mask1 = TF.center_crop(mask1_proj, [crop_H, crop_W])
            mask1_dict[level] = mask1

            g2s2_feat = TF.center_crop(grd2_feat_proj, [crop_H, crop_W])
            g2s2_conf = TF.center_crop(grd2_conf_proj, [crop_H, crop_W])
            mask2 = TF.center_crop(mask2_proj, [crop_H, crop_W])
            mask2_dict[level] = mask2

            g2s1_feat_dict[level] = g2s1_feat
            g2s1_conf_dict[level] = g2s1_conf

            g2s2_feat_dict[level] = g2s2_feat
            g2s2_conf_dict[level] = g2s2_conf

        return sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, g2s2_feat_dict, g2s2_conf_dict, mask1_dict, mask2_dict

    def forward_1grd(self, sat_img, pano1, ones1, meter_per_pixel):

        sat_img = 2 * (sat_img / 255.0) - 1.0
        pano1_img = 2 * (pano1 / 255.0) - 1.0

        # sat_img = sat_img / 255.0
        # pano1_img = pano1 / 255.0

        sat_img = sat_img.contiguous()
        pano1_img = pano1_img.contiguous()

        pano1_img = pano1_img.permute(0, 3, 1, 2)

        sat_feat_dict, sat_conf_dict = self.sat_VGG(sat_img)
        pano1_feat_dict, pano1_conf_dict = self.grd_VGG(pano1_img)

        g2s1_feat_dict = {}
        g2s1_conf_dict = {}
        mask1_dict = {}
        # corr_maps = {}
        B = sat_conf_dict[0].shape[0]

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_img.device)
        mask1 = None
        if ones1 is not None:
            ones1 = ones1.cpu().numpy()

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            pano1_feat = pano1_feat_dict[level]
            pano1_conf = pano1_conf_dict[level]

            B, c, h, w = pano1_feat.shape
            A = sat_feat.shape[-1]
            crop_H = int(A * 0.4)
            crop_W = int(A * 0.4)

            if ones1 is not None:
                resized_batch1 = []

                for i in range(B):  # 遍历 batch
                    resized_image1 = cv2.resize(ones1[i], (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_batch1.append(resized_image1)

                # 将结果转回 NumPy 数组，然后再转回 PyTorch 张量
                resized_batch1 = np.stack(resized_batch1)  # 将所有图片拼接成一个数组
                mask1 = torch.from_numpy(resized_batch1)
                mask1 = mask1.to(sat_feat.device).permute(0, 3, 1, 2)
                mask1_proj, _, grd_uv = self.project_grd_to_map(
                    mask1, pano1_conf, None, shift_u, shift_v, level, meter_per_pixel)
                mask1 = TF.center_crop(mask1_proj, [crop_H, crop_W])
                mask1_dict[level] = mask1

            grd1_feat_proj, grd1_conf_proj, grd_uv = self.project_grd_to_map(
                pano1_feat, pano1_conf, None, shift_u, shift_v, level, meter_per_pixel)

            g2s1_feat = TF.center_crop(grd1_feat_proj, [crop_H, crop_W])
            g2s1_conf = TF.center_crop(grd1_conf_proj, [crop_H, crop_W])

            g2s1_feat_dict[level] = g2s1_feat
            g2s1_conf_dict[level] = g2s1_conf

        return sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, mask1_dict

    def forward(self, sat_img, pano1, ones1, pano2, ones2, meter_per_pixel):
        if pano2 is not None:
            return self.forward_2grd(sat_img, pano1, ones1, pano2, ones2, meter_per_pixel)
        else:
            return self.forward_1grd(sat_img, pano1, ones1, meter_per_pixel)

    def calc_corr_for_train(self, sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask_dict=None,
                            batch_wise=False):
        corr_maps = {}

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            # sat_conf = sat_conf_dict[level]
            bev_feat = bev_feat_dict[level]
            bev_conf = bev_conf_dict[level]
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
        bev_conf = bev_conf_dict[level]
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
