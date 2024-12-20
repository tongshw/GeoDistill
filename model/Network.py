import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models

from model.efficientnet_pytorch import EfficientNet

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
        self.args.rotation_range = 0


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

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3-level)
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



    def forward(self, sat_img, bev_img):

        sat_img = 2 * (sat_img / 255.0) - 1.0
        bev_img = 2 * (bev_img / 255.0) - 1.0
        sat_img = sat_img.contiguous()
        bev_img = bev_img.contiguous()

        sat_feat_dict, sat_conf_dict = self.sat_VGG(sat_img)
        bev_feat_dict, bev_conf_dict = self.grd_VGG(bev_img)

        # bev_feat_dict = {}
        # bev_conf_dict = {}
        # corr_maps = {}

        # for _, level in enumerate(self.levels):
        #     sat_feat = sat_feat_dict[level]
        #     bev_feat = grd_feat_dict[level]
        #     bev_conf = grd_conf_dict[level]
        #
        #     A = sat_feat.shape[-1]

            # crop_H = int(A * 0.4)
            # crop_W = int(A * 0.4)
            # bev_feat = TF.center_crop(bev_feat, [crop_H, crop_W])
            # bev_conf = TF.center_crop(bev_conf, [crop_H, crop_W])

            # bev_feat_dict[level] = bev_feat
            # bev_conf_dict[level] = bev_conf
            #
            # B = bev_feat.shape[0]
            #
            # signal = sat_feat.repeat(1, B, 1, 1)  # [B(M), BC(NC), H, W] [8, 2048, 64, 64]
            # kernel = bev_feat * bev_conf.pow(2)  # [8, 256, 25, 25]
            # corr = F.conv2d(signal, kernel, groups=B)  # [8, 8, 40, 40], B=8
            #
            # # denominator
            # denominator_sat = []
            # sat_feat_pow = (sat_feat).pow(2)
            # bev_conf_pow = bev_conf.pow(2)
            # for i in range(0, B):
            #     denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], bev_conf_pow), dim=0)
            #     denominator_sat.append(denom_sat)
            # denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  # [B (M), B (N), H, W]
            #
            # denom_grd = torch.linalg.norm((bev_feat * bev_conf).reshape(B, -1), dim=-1)  # [B]
            # shape = denominator_sat.shape
            # denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])
            #
            # denominator = denominator_sat * denominator_grd
            #
            # denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)
            #
            # corr = 2 - 2 * corr / denominator  # [B, B, H, W]
            #
            # corr_maps[level] = corr
            #


        return sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict

    def calc_corr_for_train(self, sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict):
        corr_maps = {}

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            sat_conf = sat_conf_dict[level]
            bev_feat = bev_feat_dict[level]
            bev_conf = bev_conf_dict[level]

            B = bev_feat.shape[0]

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

            corr = 2 - 2 * corr / denominator  # [B, B, H, W]

            corr_maps[level] = corr


        return corr_maps

    def calc_corr_for_val(self, sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict):
        level = self.levels[-1]

        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        bev_feat = bev_feat_dict[level]
        bev_conf = bev_conf_dict[level]

        B, C, crop_H, crop_W = bev_feat.shape
        A = sat_feat.shape[2]

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
