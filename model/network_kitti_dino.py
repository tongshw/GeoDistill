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

from model.dpt import DPT
from model.efficientnet_pytorch import EfficientNet
from torchvision import transforms
from VGG import VGGUnet
import torchvision.transforms.functional as TF


from utils.util import grid_sample, visualize_feature_map_pca, visualize_feature_map, get_meter_per_pixel, \
    get_process_satmap_sidelength, get_camera_height


class LocalizationNet(nn.Module):
    def __init__(self, args, grid_size=8):
        super().__init__()

        self.levels = args.levels
        self.channels = args.channels

        self.shift_range_lon = args.shift_range_lon
        self.shift_range_lat = args.shift_range_lat
        self.rotation_range = args.rotation_range

        input_dim = 3
        # self.sat_VGG = VGGUnet(self.levels, self.channels)
        # self.grd_VGG = VGGUnet(self.levels, self.channels) if args.p_siamese else None

        self.SatDPT = DPT(input_dims=[1024*2, 1024*2, 1024*2, 1024*2])
        self.GrdDPT = DPT(input_dims=[1024*2, 1024*2, 1024*2, 1024*2])


        feature_dim = 320
        self.rotation_range = 0
        self.grd_height = -2

        self.meters_per_pixel = {}
        meter_per_pixel = get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel[level] = meter_per_pixel * (2 ** (3 - level))


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

    def sat2world(self, satmap_sidelength):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = get_meter_per_pixel()
        meter_per_pixel *= get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap


    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH,
                             ori_grdW):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.shift_range_lon * ori_shift_u
        shift_v_meters = self.shift_range_lat * ori_shift_v
        heading = ori_heading * self.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        H, W = uv.shape[1:-1]
        assert (H == W)

        # with torch.no_grad():
        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6) * \
               torch.greater_equal(uv[:, :, :, 0:1], torch.zeros_like(uv[:, :, :, 0:1])) * \
               torch.less(uv[:, :, :, 0:1], torch.ones_like(uv[:, :, :, 0:1]) * grd_W) * \
               torch.greater_equal(uv[:, :, :, 1:2], torch.zeros_like(uv[:, :, :, 1:2])) * \
               torch.less(uv[:, :, :, 1:2], torch.ones_like(uv[:, :, :, 1:2]) * grd_H)
        uv = uv * mask

        return uv, mask
        # return uv1


    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW, require_jac=True):
        # inputs:
        #   grd_f: ground features: B,C,H,W
        #   shift: B, S, 2
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength)  # [ sidelength,sidelength,4]

        uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        # [B, H, W, 2], [B, H, W, 1]

        grd_f_trans, new_jac = grid_sample(grd_f, uv, None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        grd_f_trans = grd_f_trans * mask[:, None, :, :, 0]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
            grd_c_trans = grd_c_trans * mask[:, None, :, :, 0]
        else:
            grd_c_trans = None


        return grd_f_trans, grd_c_trans, uv, mask

    def forward(self, sat_feat_list, grd_feat_list, left_camera_k, fov_mask=None):
        '''
        rot_corr
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        ori_grdH, ori_grdW = 256, 1024

        max1 = grd_feat_list[0].max()
        max2 = grd_feat_list[1].max()
        sat_feat_dict = self.SatDPT(sat_feat_list)

        grd_feat_dict = self.GrdDPT(grd_feat_list)

        max2 = grd_feat_dict[0].max()
        min2 = grd_feat_dict[0].min()


        B = sat_feat_dict[0].shape[0]

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_feat_dict[0].device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_feat_dict[0].device)

        g2s_feat_dict = {}
        g2s_conf_dict = {}

        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_feat_dict[0].device)
        shift_lats = None
        shift_lons = None


        # grd_feat_dict_forT, grd_conf_dict_forT = self.grd_VGG(grd_img_left)
        # sat_feat_dict_forT, sat_conf_dict_forT = self.sat_VGG(sat_map)

        grd_uv_dict = {}
        mask_dict = {}
        fov_mask_dict = {}
        for _, level in enumerate(self.levels):
            # meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_dict[level]
            grd_feat = grd_feat_dict[level]
            grd_conf = torch.ones_like(grd_feat)
            _, c, h, w = grd_feat.shape

            A = sat_feat.shape[-1]
            grd_feat_proj, grd_conf_proj, grd_uv, mask = self.project_grd_to_map(
                grd_feat, grd_conf, shift_u, shift_v, heading, left_camera_k, A, ori_grdH,
                ori_grdW, require_jac=False)

            if fov_mask is not None:

                # fig = plt.figure(figsize=(20, 5), dpi=100)  # 可自定义尺寸
                # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
                # ax.axis('off')  # 不显示坐标轴
                # mask = fov_mask[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
                # ax.imshow(mask.astype(np.uint8))
                # plt.show()

                resized_mask_batch = []

                for i in range(B):  # 遍历 batch
                    resized_mask = cv2.resize(fov_mask[i].detach().cpu().numpy().transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
                    resized_mask_batch.append(resized_mask)

                # fig = plt.figure(figsize=(10, 5), dpi=100)  # 可自定义尺寸
                # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
                # ax.axis('off')  # 不显示坐标轴
                # ax.imshow(resized_batch1[0])
                # plt.show()

                # 将结果转回 NumPy 数组，然后再转回 PyTorch 张量
                resized_mask_batch = np.stack(resized_mask_batch)  # 将所有图片拼接成一个数组
                mask_ = torch.from_numpy(resized_mask_batch)
                mask_ = mask_.to(sat_feat.device).permute(0, 3, 1, 2)

                fov_mask_proj, _, _, _ = self.project_grd_to_map(
                    mask_, grd_conf, shift_u, shift_v, heading, left_camera_k, A, ori_grdH,
                    ori_grdW, require_jac=False)
                fov_mask_dict[level] = fov_mask_proj
                # fig = plt.figure(figsize=(10, 5), dpi=100)  # 可自定义尺寸
                # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
                # ax.axis('off')  # 不显示坐标轴
                # ax.imshow(fov_mask[0])
                # plt.show()

            # grd_proj, grd_proj, _, mask = self.project_grd_to_map(
            #     ori_grd, ori_grd, shift_u, shift_v, heading, left_camera_k, A, ori_grdH,
            #     ori_grdW,
            #     require_jac=False)
            # fig = plt.figure(figsize=(10, 5), dpi=100)  # 可自定义尺寸
            # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
            # ax.axis('off')  # 不显示坐标轴
            # ax.imshow(grd_proj[0].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()

            g2s_feat_dict[level] = grd_feat_proj
            g2s_conf_dict[level] = grd_conf_proj
            grd_uv_dict[level] = grd_uv
            mask_dict[level] = mask


        for _, level in enumerate(self.levels):

            meter_per_pixel = self.meters_per_pixel[level]
            sat_feat = sat_feat_dict[level]

            A = sat_feat.shape[-1]

            crop_H = int(A - 20 * 3 / meter_per_pixel)
            crop_W = int(A - 20 * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(g2s_feat_dict[level], [crop_H, crop_W])

            g2s_conf = TF.center_crop(g2s_conf_dict[level], [crop_H, crop_W])

            g2s_feat_dict[level] = g2s_feat
            g2s_conf_dict[level] = g2s_conf
            if fov_mask is not None:
                g2s_mask = TF.center_crop(fov_mask_dict[level], [crop_H, crop_W])
                fov_mask_dict[level] = g2s_mask

        return sat_feat_dict, g2s_feat_dict, fov_mask_dict

    def calc_corr_for_train(self, sat_feat_dict, bev_feat_dict, mask_dict=None,
                            batch_wise=False):
        corr_maps = {}

        for _, level in enumerate(self.levels):
            sat_feat = sat_feat_dict[level]
            # sat_conf = sat_conf_dict[level]
            bev_feat = bev_feat_dict[level]
            bev_conf = torch.ones_like(bev_feat)[:, :1, :, :]
            if mask_dict is not None:
                mask = mask_dict[level]

                # fig = plt.figure(figsize=(5, 5), dpi=100)  # 可自定义尺寸
                # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
                # ax.axis('off')  # 不显示坐标轴
                # mask1 = mask[0].detach().cpu().numpy().transpose(1, 2, 0)
                # ax.imshow(mask1)
                # plt.show()


                mask = mask[:, 0, :, :]
                mask = mask.unsqueeze(1)

                bev_conf = bev_conf * mask
                mask = mask.repeat(1, bev_feat.shape[1], 1, 1)
                bev_feat = bev_feat * mask
            B = bev_feat.shape[0]
            A = sat_feat.shape[2]

            if batch_wise:
                max_ = bev_feat.max()
                min_ = bev_feat.min()
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

    def calc_corr_for_val(self, sat_feat_dict, bev_feat_dict, mask_dict=None):
        level = self.levels[-1]

        sat_feat = sat_feat_dict[level]
        bev_feat = bev_feat_dict[level]
        bev_conf = torch.ones_like(bev_feat)[:, :1, :, :]
        mask = None
        if mask_dict is not None:
            mask = mask_dict[level]

        B, C, crop_H, crop_W = bev_feat.shape
        A = sat_feat.shape[2]

        if mask is not None:
            mask = mask[:, 0, :, :]
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
