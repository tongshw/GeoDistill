import argparse
import json
import os

import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import cv2
import numpy as np
import torch
from easydict import EasyDict

from main import load_trained_model
from model.Network import LocalizationNet
from utils.util import vis_corr
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
def validate_single(args, model, sat, grd, sat_delta, meter_per_pixel, file_name, vis=False):
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
            pano1_feat_dict = model(sat, grd, None, None, None, meter_per_pixel)

        corr = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, None)

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
        gt_x, gt_y = gt_points[:, 0], gt_points[:, 1]

        distance = np.sqrt((gt_x.cpu().numpy() - pred_x.cpu().numpy()) ** 2 + (gt_y.cpu().numpy() - pred_y.cpu().numpy()) ** 2)
        for i in range(B):
            if vis:
                save_path = f"./vis/uncertainty/{args.model_name}/{file_names[i].cpu().numpy()}/{i}.png"
                vis_corr(corr[i], sat[i], grd[i], gt_points[i], [pred_x[i], pred_y[i]], save_path)
            else:
                vis_corr(corr[i], sat[i], grd[i], gt_points[i], [pred_x[i], pred_y[i]], None)

    return np.sqrt((gt_x.cpu().numpy() - pred_x.cpu().numpy()) ** 2 + (gt_y.cpu().numpy() - pred_y.cpu().numpy()) ** 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--img_path', default="/data/dataset/CVUSA/bingmap/20", type=str, help="path of config file")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="geoalign-CVUSA", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
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
    if config['wandb']:
        wandb.init(project="g2s-distillation", name=args.name, config=config)
    print(config)
    args = config
    sats = []
    panos = []
    deltas = []
    meter_per_pixels = []
    file_names = []

    device = torch.device("cuda:" + str(args.gpuid[0]))

    model = LocalizationNet(args).to(device)
    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)

    meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                            'Seattle': 0.100817 * 640 / 512,
                            'SanFrancisco': 0.118141 * 640 / 512,
                            'Chicago': 0.111262 * 640 / 512}
    df = pd.read_csv('/data/dataset/CVUSA/splits/val-19zl.csv', header=None)

    # 遍历文件夹中的所有图片
    count = 0
    meter_per_pixel = 0.113248 * 640 / 512
    distances = []
    # for index, row in df.iterrows(): # pano path
    for filename in os.listdir(args.img_path):
        count += 1
        # sat, pano = row[0], row[1]
        # pano_path = os.path.join(args.img_path, pano)
        # sat_path = os.path.join(args.img_path, sat)

        sat_path = os.path.join(args.img_path, filename)
        parts = sat_path.split('/')
        parts[-2] = 'panos'
        parts[-3] = 'streetview'
        # parts[-1] = parts[-1].replace('jpg', 'png')
        # 重新将列表连接回字符串
        pano_path = '/'.join(parts)


        file_names.append(sat_path.split('/')[-1].split('.')[0])
        patch_size = 512

        x, y = 0, 0

        sat = cv2.imread(sat_path, 1)[:, :, ::-1]
        height, width = sat.shape[:2]
        start_x = (width - 640) // 2
        start_y = (height - 640) // 2
        cropped_sat = sat[start_y:start_y + 640, start_x:start_x + 640]
        sat = cv2.resize(cropped_sat, (patch_size, patch_size))


        pano = cv2.imread(pano_path, 1)[:, :, ::-1]
        top_padding = (616 - pano.shape[0]) // 2  # 计算上方填充的黑色区域
        bottom_padding = 616 - pano.shape[0] - top_padding  # 计算下方填充的黑色区域
        pano_with_padding = np.pad(pano, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode='constant',
                                   constant_values=0)
        resized_pano = cv2.resize(pano_with_padding, (640, 320))


        h, w, c = resized_pano.shape
        mask_fov = 360 - args.fov_size
        mask_step = args.fov_size / args.batch_size

        sats.append(torch.from_numpy(sat.copy()).float().permute(2, 0, 1))
        panos.append(torch.from_numpy(resized_pano.copy()).float())
        deltas.append([x, y])
        meter_per_pixels.append(meter_per_pixel)

        if count % args.batch_size == 0:
            sat_batch = torch.stack(sats).to(device)
            pano_batch = torch.stack(panos).to(device)
            delta = torch.tensor(deltas).to(device)
            meter_per_pixel_batch = torch.tensor(meter_per_pixels).to(device)
            distance = validate_single(config, model, sat_batch, pano_batch, delta, meter_per_pixel_batch, file_names)
            distances.append(distance)

            sats = []
            panos = []
            deltas = []
            meter_per_pixels = []
            file_names = []
            # print(f"batch {count // args.batch_size}")


    all_distances = np.concatenate(distances, axis=0)
    mean_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)

    print('mean distance（pixels):', mean_distance)
    print('median distance（pixels):', median_distance)
    if config['wandb']:
        wandb.finish()