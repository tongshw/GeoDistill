import argparse
import copy
import json
import os

import cv2
import math
import numpy as np
from matplotlib.colors import Normalize

from model.dino import DINO
from model.orientation_estimator import RotationPredictionNet

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['WANDB_MODE'] = "offline"
import time

import torch
import wandb
from easydict import EasyDict
from matplotlib import pyplot as plt, gridspec
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.network_vigor import LocalizationNet

from utils.util import setup_seed, print_colored, count_parameters, visualization, TextColors, vis_corr, vis_two_sat, \
    visualize_distributions
from dataset.VIGOR import fetch_dataloader, VIGOR

import torch.nn.functional as F


def load_trained_model(model, pth_file, device):
    # 加载保存的模型权重
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {pth_file}, epoch {checkpoint['epoch']}, val_loss {checkpoint['loss']:.4f}")
    return model, checkpoint['epoch'], checkpoint['loss']


def validate_with_rotation(args, dino, model, rotation_estimator,  val_loader, criterion, device, epoch=-1, vis=False, name=None):
    model.eval()
    rotation_estimator.eval()
    total_loss = 0
    all_errors = []

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    rotation_errs = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano, city, \
                masked_fov, ori_angle = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
            city = data_blob[-1]

            # plt.figure(figsize=(10, 5))
            # plt.imshow(pano1[0].cpu().detach().numpy().astype(np.uint8))
            # plt.show()
            #
            pred_label = rotation_estimator(sat, bev)
            predicted_labels = torch.argmax(pred_label, dim=1)
            predicted_labels -= args.ori_noise
            for i in range(resized_pano.shape[0]):
                # print(f"after shift:{shift}")
                # print(resized_pano.shape)
                # 反向移位
                resized_pano_cpu = resized_pano[i].cpu().numpy()
                resized_pano_cpu = np.roll(resized_pano_cpu, -int(predicted_labels[i].cpu().detach().numpy() / 360 * 640), axis=1)
                resized_pano[i] = torch.tensor(resized_pano_cpu).to(resized_pano.device)

            # pred_label = rotation_estimator(sat, bev)
            # pred_ori = torch.argmax(pred_label, dim=1) - 45
            #
            # err, mean_err, median_err = calculate_errors(ori_angle, pred_label)
            # rotation_errs.extend(err.cpu().numpy())
            #
            # # print(f"pred:{pred_ori[0]}, gt_ori:{ori_angle[0]}")
            #
            # for i in range(resized_pano.shape[0]):
            #     if pred_ori[i] > 0:
            #         shift = round(pred_ori[i].cpu().numpy() * 640 / 360)
            #     else:
            #         shift = math.floor(pred_ori[i].cpu().numpy() * 640 / 360)
            #     # print(f"after shift:{shift}")
            #     # print(resized_pano.shape)
            #     # 反向移位
            #     resized_pano_cpu = resized_pano[i].cpu().numpy()
            #     resized_pano_cpu = np.roll(resized_pano_cpu, -shift, axis=1)
            #
            #     # 如果需要将结果返回到 GPU
            #     resized_pano[i] = torch.tensor(resized_pano_cpu).to(resized_pano.device)

            # plt.figure(figsize=(10, 5))
            # plt.imshow(resized_pano[0].cpu().detach().numpy().astype(np.uint8))
            # plt.show()


            sat_img = 2 * (sat / 255.0) - 1.0
            pano_img = 2 * (resized_pano / 255.0) - 1.0

            sat_img = sat_img.contiguous()
            pano_img = pano_img.contiguous()

            pano_img = pano_img.permute(0, 3, 1, 2)

            sat_feat_list = dino(sat_img)
            pano_feat_list = dino(pano_img)

            sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict, pano1_conf_dict,\
                pano1_feat_dict = model(sat_feat_list, pano_feat_list, meter_per_pixel, mask=None)

            corr = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask_dict=None)

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
            if i_batch % 25 == 0 and args.visualize:
                if args.save_visualization:
                    if epoch == -1:
                        save_path = f"./vis/distillation/{args.model_name}/test/{sat_gps[0].cpu().numpy()}.png"
                    else:
                        if name is None:
                            save_path = f"./vis/distillation/{args.model_name}/val/{epoch}/{sat_gps[0].cpu().numpy()}.png"
                        else:
                            save_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/{sat_gps[0].cpu().numpy()}.png"
                    vis_corr(corr[0], sat[0], pano1[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)
                else:
                    vis_corr(corr[0], sat[0], pano1[0], gt_points[0], [pred_x[0], pred_y[0]], None)


    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)

    metrics = [1, 3, 5]
    mean_dis = np.mean(distance)
    median_dis = np.median(distance)
    overall_median = np.median(rotation_errs)
    overall_mean = np.mean(rotation_errs)
    if args.wandb:
        if name is not None:
            wandb.log({f'val_{name}_mean': mean_dis,
                       f'val_{name}_median': median_dis, })
        else:
            wandb.log({'val_mean': mean_dis,
                   'val_median': median_dis, })


    print(f"mean distance: {mean_dis:.4f}")
    print(f"median distance: {median_dis:.4f}")
    print(f"mean rotation: {overall_mean:.4f}")
    print(f"median rotation: {overall_median:.4f}")

    return mean_dis, median_dis


def test_with_rotation(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    model = LocalizationNet(args).to(device)

    rotation_estimator = RotationPredictionNet(args, num_classes=args.ori_noise * 2).to(device)
    rotation_estimator, start_epoch, best_val_loss = load_trained_model(rotation_estimator, args.orientation_model, device)
    dinov2 = DINO().to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    criterion = nn.CrossEntropyLoss()
    mean_error, median_error = validate_with_rotation(args, dinov2, model, rotation_estimator, test_loader, criterion, device, name="student")

    print(f"mean rotation: {mean_error:.4f}")
    print(f"median rotation: {median_error:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config_vigor.json", type=str, help="path of config file")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="cross-train_on_NY-3dof-infer", help="none")
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese


    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['name'] = args.name
    # config['restore_ckpt'] = args.restore_ckpt
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if config['wandb']:
        wandb.init(project="g2s-distillation", name=args.name, config=config)


    print(config)

    start_time = time.strftime('%Y%m%d_%H%M%S')
    config['model_name'] = f"{args.name}_{start_time}"
    print(f"model_name: {config.model_name}")

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    test_with_rotation(config)

    if config['wandb']:
        wandb.finish()
