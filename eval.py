import argparse
import json
import os

from eval_uncertainty import calculate_entropy

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import cv2
import numpy as np
import torch
from easydict import EasyDict

from main import load_trained_model
from model.Network import LocalizationNet
from utils.util import vis_corr
from PIL import Image
import torch.nn.functional as F

def validate_single(args, model, sat, grd, mask, sat_delta, sat_gps, meter_per_pixel, vis=False):
    model.eval()
    total_loss = 0
    all_errors = []

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    torch.cuda.empty_cache()
    with torch.no_grad():

        # 前向传播
        sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict, pano1_conf_dict, \
            pano1_feat_dict = model(sat, grd, mask, None, None, meter_per_pixel)

        corr = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict)

        # # 计算损失
        # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        # loss = 100 * cls_loss + 1 * reg_loss

        max_level = args.levels[-1]

        B, corr_H, corr_W = corr.shape

        max_index = torch.argmax(corr.reshape(B, -1), dim=1)
        pred_u = (max_index % corr_W - corr_W / 2)
        pred_v = (max_index // corr_W - corr_H / 2)

        pred_u = pred_u * np.power(2, 3 - max_level) * meter_per_pixel
        pred_v = pred_v * np.power(2, 3 - max_level) * meter_per_pixel

        pred_us.append(pred_u.data.cpu().numpy())
        pred_vs.append(pred_v.data.cpu().numpy())

        gt_shift_u = sat_delta[:, 0] * meter_per_pixel * 512 / 4
        gt_shift_v = sat_delta[:, 1] * meter_per_pixel * 512 / 4

        gt_us.append(gt_shift_u.data.cpu().numpy())
        gt_vs.append(gt_shift_v.data.cpu().numpy())

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]
        pred_x = pred_u / meter_per_pixel + 512 / 2
        pred_y = pred_v / meter_per_pixel + 512 / 2



        for i in range(B):
            corr_map1_flat = corr[i].view(-1)  # 展平成向量，形状为 (h*w,)

            # 对展平的相关性矩阵进行 softmax
            corr_map1_softmax = F.softmax(corr_map1_flat / 0.5, dim=0)  # 按所有元素求 softmax

            # 将 softmax 结果 reshape 回原来的 (h, w) 形状
            corr_map1 = corr_map1_softmax.view(corr_H, corr_W)

            print(f"uncertainty:{calculate_entropy(corr_map1[i])}")
            if vis:
                save_path = f"./vis/uncertainty/{args.model_name}/{sat_gps[i].cpu().numpy()}/{i}.png"
                vis_corr(corr[i], sat[i], grd[i], gt_points[i], [pred_x[i], pred_y[i]], save_path, temp=0.06)
            else:
                vis_corr(corr[i], sat[i], grd[i], gt_points[i], [pred_x[i], pred_y[i]], None, temp=0.06)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--img_path', default="/data/test/code/multi-local/image/better/4/pano", type=str, help="path of config file")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="same-2subview", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=False, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=False)

    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['validation'] = args.validation
    config['name'] = args.name
    config['img_path'] = args.img_path
    # config['restore_ckpt'] = args.restore_ckpt
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    config['train'] = args.train
    config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    args = config
    sats = []
    panos = []
    masks = []
    deltas = []
    meter_per_pixels = []

    device = torch.device("cuda:" + str(args.gpuid[0]))

    model = LocalizationNet(args).to(device)
    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)

    meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                            'Seattle': 0.100817 * 640 / 512,
                            'SanFrancisco': 0.118141 * 640 / 512,
                            'Chicago': 0.111262 * 640 / 512}

    # 遍历文件夹中的所有图片
    for filename in os.listdir(args.img_path): # pano path
        # if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据文件类型选择
        pano_path = os.path.join(args.img_path, filename)
        parts = pano_path.split('/')
        parts[-2] = 'sat'
        parts[-1] = parts[-1].replace('jpg', 'png')
        # 重新将列表连接回字符串
        sat_path = '/'.join(parts)

        patch_size = 512


        city = sat_path.split('/')[-1].split('_')[0]
        part = sat_path.split('/')[-1].split('_')[1]
        x, y = map(float, part.split(','))

        sat = cv2.imread(sat_path, 1)[:, :, ::-1]
        sat = cv2.resize(sat, (patch_size, patch_size))

        pano = cv2.imread(pano_path, 1)[:, :, ::-1]
        resized_pano = cv2.resize(pano, (640, 320))

        h, w, c = resized_pano.shape
        count = 0
        mask_fov = 360 - args.fov_size
        # mask_step = args.fov_size / args.batch_size
        mask_step = 40
        for i in range(args.batch_size):
            # fov=240
            start_angle = count * mask_step
            w_start1 = int(np.round(w / 360 * start_angle))
            w_end1 = int(np.round(w / 360 * (start_angle + mask_fov)))
            count += 1

            # 创建两个 mask，并应用到原图像上
            mask = np.zeros_like(resized_pano)
            pano = resized_pano.copy()
            ones = np.ones_like(resized_pano)

            pano[:, w_start1:w_end1, :] = mask[:, w_start1:w_end1, :]
            ones[:, w_start1:w_end1, :] = mask[:, w_start1:w_end1, :]

            panos.append(torch.from_numpy(pano).float())
            masks.append(torch.from_numpy(ones).float())
            sats.append(torch.from_numpy(sat).float().permute(2, 0, 1))
            deltas.append([x, y])
            meter_per_pixels.append(meter_per_pixel_dict[city])

        sat_batch = torch.stack(sats).to(device)
        pano_batch = torch.stack(panos).to(device)
        mask_batch = torch.stack(masks).to(device)
        delta = torch.tensor(deltas).to(device)
        meter_per_pixel = torch.tensor(meter_per_pixels).to(device)
        validate_single(config, model, sat_batch, pano_batch, mask_batch, delta, None, meter_per_pixel)
        sats = []
        panos = []
        masks = []
        deltas = []
        meter_per_pixels = []
