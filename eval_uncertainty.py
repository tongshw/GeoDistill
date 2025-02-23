import argparse
import copy
import json
import os

import cv2
import numpy as np
from matplotlib.colors import Normalize
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['WANDB_MODE'] = "offline"
import time
from PIL import Image
import torch
import wandb
from easydict import EasyDict
from matplotlib import pyplot as plt, gridspec
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Network import LocalizationNet
from utils.util import setup_seed, print_colored, count_parameters, visualization, TextColors, vis_corr, vis_two_sat, \
    visualize_distributions
from dataset.VIGOR import fetch_dataloader, VIGOR


def load_trained_model(model, pth_file, device):
    # 加载保存的模型权重
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {pth_file}, epoch {checkpoint['epoch']}, val_loss {checkpoint['loss']:.4f}")
    return model, checkpoint['epoch'], checkpoint['loss']


def calculate_entropy(similarity_map):
    """
    Calculate the entropy of a similarity map with values in the range [-1, 1].

    Parameters:
    similarity_map (torch.Tensor or numpy.ndarray): 2D array of similarity values.

    Returns:
    float: The entropy of the similarity map.
    """
    # If the similarity_map is a torch tensor, move it to CPU and convert to numpy
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = similarity_map.cpu().numpy()

    # # Normalize similarity map to the range [0, 1]
    # similarity_map = (similarity_map + 1) / 2  # This shifts the range from [-1, 1] to [0, 1]

    # Flatten the similarity map to a 1D array
    flattened_map = similarity_map.flatten()

    # Compute the histogram of the similarity values
    hist, bin_edges = np.histogram(flattened_map, bins=50, range=(0, 1), density=True)

    # Avoid log(0) by replacing 0s with a small value (epsilon)
    hist = np.clip(hist, a_min=1e-10, a_max=None)
    hist /= np.sum(hist)
    # Compute entropy
    entropy = -np.sum(hist * np.log(hist))

    return entropy


def compare_batch_uncertainty(batch):
    """
    Compare the entropy (uncertainty) of a batch of similarity maps.

    Parameters:
    batch (numpy.ndarray): 3D array of similarity maps with shape [batch, h, w].

    Returns:
    numpy.ndarray: Array of entropy values for each similarity map in the batch.
    """
    entropies = []

    # Iterate over the batch and calculate entropy for each similarity map
    for i in range(batch.shape[0]):
        entropy = calculate_entropy(batch[i])
        entropies.append(entropy)

    entropies = np.array(entropies)

    # Find the index of the map with the lowest entropy (least uncertainty)
    min_entropy_index = np.argmin(entropies)

    print(f"Entropies of the batch maps: {entropies}")
    print(f"Map with the least uncertainty (lowest entropy) is at index {min_entropy_index}")

    return min_entropy_index, entropies




def validate_distillation(args, model, val_loader, criterion, device, epoch=-1, vis=False, name=None, model_s = None):
    model.eval()
    if model_s is not None:
        model_s.eval()
    total_loss = 0
    all_errors = []

    pred_us_t = []
    pred_vs_t = []

    gt_us = []
    gt_vs = []

    pred_us_s = []
    pred_vs_s = []

    sat_dir = f"./image/{args.model_name}/sat/"
    pano_dir = f"./image/{args.model_name}/pano/"

    # 确保目录存在
    os.makedirs(sat_dir, exist_ok=True)
    os.makedirs(pano_dir, exist_ok=True)

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano, city, masked_fov = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
            city = data_blob[-1]

            # 前向传播
            sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, g2s2_feat_dict, g2s2_conf_dict, mask1_dict, mask2_dict, \
                pano1_conf_dict, pano2_conf_dict = model(sat, resized_pano, ones2, pano1, ones1, meter_per_pixel)

            corr_t = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, None)
            corr_s = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, g2s2_feat_dict, g2s2_conf_dict, mask2_dict)

            if model_s is not None:
                sat_feat_dict_t, sat_conf_dict_t, bev_feat_dict_t, bev_conf_dict_t, mask1_dict_t, pano1_conf_dict_t, \
                    pano1_feat_dict_t = model(sat, resized_pano, None,None, None, meter_per_pixel)

                corr_t = model.calc_corr_for_val(sat_feat_dict_t, sat_conf_dict_t, bev_feat_dict_t, bev_conf_dict_t,
                                                   None)

                sat_feat_dict_s, sat_conf_dict_s, bev_feat_dict_s, bev_conf_dict_s, mask1_dict_s, pano1_conf_dict_s, \
                    pano1_feat_dict_s = model_s(sat, resized_pano, None,None, None, meter_per_pixel)

                corr_s = model_s.calc_corr_for_val(sat_feat_dict_s, sat_conf_dict_s, bev_feat_dict_s, bev_conf_dict_s,
                                                   None)

            # # 计算损失
            # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
            # loss = 100 * cls_loss + 1 * reg_loss

            max_level = args.levels[-1]

            B, corr_H, corr_W = corr_t.shape

            # teacher or before
            max_value_t, max_index_t = torch.max(corr_t.reshape(B, -1), dim=1)
            pred_u_t = (max_index_t % corr_W - corr_W / 2)
            pred_v_t = (max_index_t // corr_W - corr_H / 2)

            pred_u_t = pred_u_t * np.power(2, 3 - max_level)
            pred_u_t *= meter_per_pixel
            pred_v_t = pred_v_t * np.power(2, 3 - max_level) * meter_per_pixel

            pred_us_t.append(pred_u_t.data.cpu().numpy())
            pred_vs_t.append(pred_v_t.data.cpu().numpy())

            # student
            max_value_s, max_index_s = torch.max(corr_s.reshape(B, -1), dim=1)
            pred_u_s = (max_index_s % corr_W - corr_W / 2)
            pred_v_s = (max_index_s // corr_W - corr_H / 2)

            pred_u_s = pred_u_s * np.power(2, 3 - max_level) * meter_per_pixel
            pred_v_s = pred_v_s * np.power(2, 3 - max_level) * meter_per_pixel

            pred_us_s.append(pred_u_s.data.cpu().numpy())
            pred_vs_s.append(pred_v_s.data.cpu().numpy())


            gt_shift_u = sat_delta[:, 0] * meter_per_pixel * 512 / 4
            gt_shift_v = sat_delta[:, 1] * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            gt_points = sat_delta * 512 / 4
            gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
            gt_points[:, 1] = 512 / 2 + gt_points[:, 1]
            pred_x_t = pred_u_t / meter_per_pixel + 512 / 2
            pred_y_t = pred_v_t / meter_per_pixel + 512 / 2

            pred_x_s = pred_u_s / meter_per_pixel + 512 / 2
            pred_y_s = pred_v_s / meter_per_pixel + 512 / 2
            distance_t = np.sqrt((pred_x_t.detach().cpu().numpy() - gt_points[:, 0].detach().cpu().numpy()) ** 2 + (pred_y_t.detach().cpu().numpy() - gt_points[:, 1].detach().cpu().numpy()) ** 2)  # [N]
            distance_s = np.sqrt((pred_x_s.detach().cpu().numpy() - gt_points[:, 0].detach().cpu().numpy()) ** 2 + (pred_y_s.detach().cpu().numpy() - gt_points[:, 1].detach().cpu().numpy()) ** 2)  # [N]

            for i in range(B):
                # corr_map1_flat = corr_t[i].view(-1)  # 展平成向量，形状为 (h*w,)
                # corr_map2_flat = corr_s[i].view(-1)  # 展平成向量，形状为 (h*w,)
                # # max_corr1, _ = torch.max(corr_map1_flat, dim=0)
                # # max_corr2, _ = torch.max(corr_map2_flat, dim=0)
                #
                # # 对展平的相关性矩阵进行 softmax
                # corr_map1_softmax = F.softmax(corr_map1_flat / 0.06, dim=0)  # 按所有元素求 softmax
                # corr_map2_softmax = F.softmax(corr_map2_flat / 0.06, dim=0)  # 按所有元素求 softmax
                # # max1 = torch.max(corr_map1_softmax, dim=0)
                # # max2 = torch.max(corr_map2_softmax, dim=0)
                #
                #
                # # 将 softmax 结果 reshape 回原来的 (h, w) 形状
                # corr_map1 = corr_map1_softmax.view(corr_H, corr_W)
                # corr_map2 = corr_map2_softmax.view(corr_H, corr_W)
                #
                # entropy_t = calculate_entropy(corr_map1)
                # entropy_s = calculate_entropy(corr_map2)

                if distance_t[i] > distance_s[i]:
                    save_path1 = None
                    save_path2 = None
                    if args.save_visualization:
                        gps = sat_gps[i].cpu().numpy()
                        delta_x, delta_y = sat_delta[i][0].cpu().numpy().item(), sat_delta[i][1].cpu().numpy().item()
                        save_path1 = f"./vis/uncertainty/{args.model_name}/before/{city[i]}_{delta_x},{delta_y}_{gps}.png"
                        save_path2 = f"./vis/uncertainty/{args.model_name}/after/{city[i]}_{delta_x},{delta_y}_{gps}.png"\

                        # sat_path = f"./image/{args.model_name}/sat/{city[i]}_{delta_x},{delta_y}_{gps}.png"
                        # pano_path = f"./image/{args.model_name}/pano/{city[i]}_{delta_x},{delta_y}_{gps}.png"
                        # sat_img = Image.fromarray(sat[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                        # sat_img.save(sat_path)
                        # pano_img = Image.fromarray(resized_pano[i].cpu().numpy().astype(np.uint8))
                        # pano_img.save(pano_path)

                    vis_corr(corr_t[i], sat[i], resized_pano[i], gt_points[i], [pred_x_t[i], pred_y_t[i]], save_path1, temp=0.1)
                    vis_corr(corr_s[i], sat[i], pano1[i], gt_points[i], [pred_x_s[i], pred_y_s[i]], save_path2, temp=0.1)


            # for i in range(B):
            #     corr_map1_flat = corr_t[i].view(-1)  # 展平成向量，形状为 (h*w,)
            #     corr_map2_flat = corr_s[i].view(-1)  # 展平成向量，形状为 (h*w,)
            #     # max_corr1, _ = torch.max(corr_map1_flat, dim=0)
            #     # max_corr2, _ = torch.max(corr_map2_flat, dim=0)
            #
            #     # 对展平的相关性矩阵进行 softmax
            #     corr_map1_softmax = F.softmax(corr_map1_flat / 0.06, dim=0)  # 按所有元素求 softmax
            #     corr_map2_softmax = F.softmax(corr_map2_flat / 0.06, dim=0)  # 按所有元素求 softmax
            #     # max1 = torch.max(corr_map1_softmax, dim=0)
            #     # max2 = torch.max(corr_map2_softmax, dim=0)
            #
            #
            #     # 将 softmax 结果 reshape 回原来的 (h, w) 形状
            #     corr_map1 = corr_map1_softmax.view(corr_H, corr_W)
            #     corr_map2 = corr_map2_softmax.view(corr_H, corr_W)
            #
            #     entropy_t = calculate_entropy(corr_map1)
            #     entropy_s = calculate_entropy(corr_map2)
            #
            #     if entropy_t > entropy_s:
            #         save_path1 = None
            #         save_path2 = None
            #         if args.save_visualization:
            #             gps = sat_gps[i].cpu().numpy()
            #             delta_x, delta_y = sat_delta[i][0].cpu().numpy().item(), sat_delta[i][1].cpu().numpy().item()
            #             save_path1 = f"./vis/uncertainty/{args.model_name}/teacher/{city[i]}_{delta_x},{delta_y}_{gps}.png"
            #             save_path2 = f"./vis/uncertainty/{args.model_name}/student/{city[i]}_{delta_x},{delta_y}_{gps}.png"\
            #
            #             sat_path = f"./image/{args.model_name}/sat/{city[i]}_{delta_x},{delta_y}_{gps}.png"
            #             pano_path = f"./image/{args.model_name}/pano/{city[i]}_{delta_x},{delta_y}_{gps}.png"
            #             sat_img = Image.fromarray(sat[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            #             sat_img.save(sat_path)
            #             pano_img = Image.fromarray(resized_pano[i].cpu().numpy().astype(np.uint8))
            #             pano_img.save(pano_path)
            #
            #         vis_corr(corr_t[i], sat[i], resized_pano[i], gt_points[i], [pred_x_t[i], pred_y_t[i]], save_path1, temp=1)
            #         vis_corr(corr_s[i], sat[i], pano1[i], gt_points[i], [pred_x_s[i], pred_y_s[i]], save_path2, temp=1)


            # if i_batch % 25 == 0 and args.visualize:
            #     if args.save_visualization:
            #         if epoch == -1:
            #             save_path = f"./vis/distillation/{args.model_name}/test/{sat_gps[0].cpu().numpy()}.png"
            #         else:
            #             if name is None:
            #                 save_path = f"./vis/distillation/{args.model_name}/val/{epoch}/{sat_gps[0].cpu().numpy()}.png"
            #             else:
            #                 save_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/{sat_gps[0].cpu().numpy()}.png"
            #         vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)
            #     else:
            #         vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], None)


    pred_us_t = np.concatenate(pred_us_t, axis=0)
    pred_vs_t = np.concatenate(pred_vs_t, axis=0)

    pred_us_s = np.concatenate(pred_us_s, axis=0)
    pred_vs_s = np.concatenate(pred_vs_s, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance_t = np.sqrt((pred_us_t - gt_us) ** 2 + (pred_vs_t - gt_vs) ** 2)  # [N]
    distance_s = np.sqrt((pred_us_s - gt_us) ** 2 + (pred_vs_s - gt_vs) ** 2)  # [N]

    count = np.sum(distance_s < distance_t)
    print(f"{count} samples where limited FoV is more accurate than 360° prediction")

    metrics = [1, 3, 5]
    mean_dis_t = np.mean(distance_t)
    median_dis_t = np.median(distance_t)

    mean_dis_s = np.mean(distance_s)
    median_dis_s = np.median(distance_s)

    if args.wandb:
        wandb.log({'val_before_mean': mean_dis_t,
                   'val_before_median': median_dis_t,
                   'val_student_mean': mean_dis_s,
                   'val_student_median': median_dis_s,
                   })


    print(f"before distillation mean distance: {mean_dis_t:.4f}")
    print(f"before distillation distance: {median_dis_t:.4f}")
    print(f"after distillation mean distance: {mean_dis_s:.4f}")
    print(f"after distillation distance: {median_dis_s:.4f}")
    return mean_dis_t, mean_dis_s, median_dis_t, median_dis_s


def test(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    model = LocalizationNet(args).to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)

    model_s = LocalizationNet(args).to(device)

    model_s, start_epoch, best_val_loss = load_trained_model(model_s, args.model1, device)

    criterion = nn.CrossEntropyLoss()
    mean_dis_t, mean_dis_s, median_dis_t, median_dis_s = validate_distillation(args, model, test_loader, criterion, device, name="student", model_s=model_s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="cross-eval_beforeVSafter", help="none")
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

    # if not config['train']:
    #     config['fov_size'] = 360
    print(config)

    start_time = time.strftime('%Y%m%d_%H%M%S')
    config['model_name'] = f"{args.name}_{start_time}"
    print(f"model_name: {config.model_name}")

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if config.dataset == 'vigor':
        print("Dataset is VIGOR!")
    if args.train:
        pass
    else:
        # pass
        test(config)
        # vis_distillation_err(config)
        # config['model'] = "/data/test/code/multi-local/location_model/cross-proj-feat-l1consistency-corr_w50_20241221_194330.pth"
        # test(config)

    if config['wandb']:
        wandb.finish()
