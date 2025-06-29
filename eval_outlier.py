import argparse
import copy
import json
import os
import random

import cv2
import numpy as np
from matplotlib.colors import Normalize

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
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

from model.Network import LocalizationNet
from model.loss import Weakly_supervised_loss_w_GPS_error, consistency_constraint_soft_L1, \
    consistency_constraint_KL_divergency, cross_entropy, generate_gaussian_heatmap, adaptive_cross_entropy, \
    cross_entropy_fully_supervised, kl_divergence, wasserstein_distance, sparse_wasserstein
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


import os
import numpy as np
import matplotlib.pyplot as plt


def validate(args, model, val_loader, criterion, device, epoch=-1, vis=False, name=None):
    model.eval()
    total_loss = 0
    all_errors = []
    all_distances = []  # 用于收集所有的distance

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    # 用字典来统计每个bin的样本数
    bin_counts = {}
    city_distances = {}

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano, city, masked_fov = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
            city = data_blob[-2]

            # 前向传播
            sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict, pano1_conf_dict, \
                pano1_feat_dict = model(sat, resized_pano, ones1, None, None, meter_per_pixel)

            corr = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, None)

            max_level = args.levels[-1]
            B, corr_H, corr_W = corr.shape

            max_index = torch.argmax(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2)
            pred_v = (max_index // corr_W - corr_H / 2)

            _, _, feat_H, feat_W = sat_feat_dict[max_level].shape

            pred_u = pred_u * 512 / feat_H * meter_per_pixel
            pred_v = pred_v * 512 / feat_H * meter_per_pixel

            gt_shift_u = sat_delta[:, 0] * meter_per_pixel * 512 / 4
            gt_shift_v = sat_delta[:, 1] * meter_per_pixel * 512 / 4

            # 计算当前batch的distance
            curr_distance = torch.sqrt((pred_u - gt_shift_u) ** 2 + (pred_v - gt_shift_v) ** 2)
            all_distances.extend(curr_distance.cpu().numpy())  # 收集所有distance

            gt_points = sat_delta * 512 / 4
            gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
            gt_points[:, 1] = 512 / 2 + gt_points[:, 1]
            pred_x = pred_u / meter_per_pixel + 512 / 2
            pred_y = pred_v / meter_per_pixel + 512 / 2

            # 对每个样本进行可视化和统计
            for b in range(B):
                curr_city = city[b]
                if curr_city not in city_distances:
                    city_distances[curr_city] = []
                city_distances[curr_city].append(curr_distance[b].cpu().numpy())

                dist = curr_distance[b].item()
                if dist > 1:
                    folder_index = int(dist)
                    folder_name = f"{folder_index}-{folder_index + 1}m"

                    # 按城市统计bin
                    if curr_city not in bin_counts:
                        bin_counts[curr_city] = {}
                    bin_counts[curr_city][folder_index] = bin_counts[curr_city].get(folder_index, 0) + 1

                    if args.save_visualization:
                        if epoch == -1:
                            base_path = f"./vis/error_analysis/{args.model_name}/{city[b]}/{folder_name}"
                        else:
                            if name is None:
                                base_path = f"./vis/distillation/{args.model_name}/val/{epoch}/error_analysis/{folder_name}"
                            else:
                                base_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/error_analysis/{folder_name}"

                        os.makedirs(base_path, exist_ok=True)
                        save_path = os.path.join(base_path, f"{sat_gps[b].cpu().numpy()}_dist{dist:.2f}.png")
                        vis_corr(corr[b], sat[b], resized_pano[b], gt_points[b], [pred_x[b], pred_y[b]], save_path)

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())
            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())


    for city_name, distances in city_distances.items():
        plt.figure(figsize=(12, 6))
        distances = np.array(distances)

        # 计算该城市小于1米的样本数量
        num_less_than_1m = np.sum(distances <= 1)
        print(f"\n{city_name} - Number of samples with distance <= 1m: {num_less_than_1m}")

        # 打印该城市的bin统计信息
        if city_name in bin_counts:
            print(f"\n{city_name} - Bin statistics:")
            for bin_index in sorted(bin_counts[city_name].keys()):
                print(f"{bin_index}-{bin_index + 1}m: {bin_counts[city_name][bin_index]} samples")

        # 绘制单个城市的直方图
        plt.hist(distances, bins=range(0, int(np.max(distances)) + 2, 1),
                 edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Localization Errors - {city_name}')
        plt.xlabel('Error Distance (meters)')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)

        # 保存单个城市的直方图
        if epoch == -1:
            hist_save_path = f"./vis/error_analysis/{args.model_name}/error_distribution_{city_name}.png"
        else:
            if name is None:
                hist_save_path = f"./vis/distillation/{args.model_name}/val/{epoch}/error_distribution_{city_name}.png"
            else:
                hist_save_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/error_distribution_{city_name}.png"

        os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
        plt.savefig(hist_save_path)
        plt.close()

    # 绘制叠加的直方图
    plt.figure(figsize=(12, 6))
    for city_name, distances in city_distances.items():
        plt.hist(distances, bins=range(0, int(max(map(np.max, city_distances.values()))) + 2, 1),
                 label=city_name, alpha=0.5)

    plt.title('Distribution of Localization Errors - All Cities')
    plt.xlabel('Error Distance (meters)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存叠加的直方图
    if epoch == -1:
        hist_save_path = f"./vis/error_analysis/{args.model_name}/error_distribution_all_cities.png"
    else:
        if name is None:
            hist_save_path = f"./vis/distillation/{args.model_name}/val/{epoch}/error_distribution_all_cities.png"
        else:
            hist_save_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/error_distribution_all_cities.png"

    os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
    plt.savefig(hist_save_path)
    plt.close()



    # 计算统计指标
    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)
    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)
    mean_dis = np.mean(distance)
    median_dis = np.median(distance)

    if args.wandb:
        if name is not None:
            wandb.log({f'val_{name}_mean': mean_dis,
                       f'val_{name}_median': median_dis})
        else:
            wandb.log({'val_mean': mean_dis,
                       'val_median': median_dis})

    print(f"\nOverall statistics:")
    print(f"Mean distance: {mean_dis:.4f}")
    print(f"Median distance: {median_dis:.4f}")

    return mean_dis, median_dis




def test(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    model = LocalizationNet(args).to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    criterion = nn.CrossEntropyLoss()
    mean_error, median_error = validate(args, model, test_loader, criterion, device, name="student")

    # print(f"Val Loss: {val_loss:.4f}")
    print(f"mean rotation: {mean_error:.4f}")
    print(f"median rotation: {median_error:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="cross-g2s-infer-s", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=False)

    parser.add_argument('--best_dis', type=float, default=1e8)

    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['best_dis'] = args.best_dis
    config['validation'] = args.validation
    config['name'] = args.name
    # config['restore_ckpt'] = args.restore_ckpt
    config['start_step'] = args.start_step
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    config['train'] = args.train
    config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if config['wandb']:
        wandb.init(project="g2s-distillation", name=args.name, config=config)

    # config['model'] = "/data/test/code/geodistill_vit/kitti/student/cross-dino-g2s-inversedataset_20250604_072506.pth"
    # config['model'] = "/data/test/code/geodistill_vit/kitti/student/cross-distill-inversedataset_20250608_192434.pth"

    config['model'] = "/data/test/code/geodistill_vit/kitti/student/cross-dino-g2sweakly_20250527_220736.pth"
    # config['model'] = "/data/test/code/geodistill_vit/kitti/student/cross-dino-distill_20250531_090106.pth"

    print(config)

    start_time = time.strftime('%Y%m%d_%H%M%S')
    config['model_name'] = f"{args.name}_{start_time}"
    print(f"model_name: {config.model_name}")

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    test(config)

    if config['wandb']:
        wandb.finish()
