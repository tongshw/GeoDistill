import argparse
import json
import os

import cv2
import numpy as np
from matplotlib.colors import Normalize

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
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
    consistency_constraint_KL_divergency
from utils.util import setup_seed, print_colored, count_parameters, visualization, TextColors, vis_corr
from dataset.VIGOR import fetch_dataloader
import torch.nn.functional as F


def load_trained_model(model, pth_file, device):
    # 加载保存的模型权重
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {pth_file}, epoch {checkpoint['epoch']}, val_loss {checkpoint['loss']:.4f}")
    return model, checkpoint['epoch'], checkpoint['loss']


def train_epoch(args, model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    # 进度条
    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        # 解包数据并移动到设备
        bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano = [
            x.to(device) for x in data_blob]


        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, \
            g2s2_feat_dict, g2s2_conf_dict, mask1, mask2 = model(sat, resized_pano, ones1, pano2, ones2, meter_per_pixel)

        corr_maps1 = model.calc_corr_for_train(sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, None)
        # corr_maps2 = model.calc_corr_for_train(sat_feat_dict, sat_conf_dict, g2s2_feat_dict, g2s2_conf_dict, mask2)

        # 计算损失
        # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        corr_loss1 = Weakly_supervised_loss_w_GPS_error(corr_maps1, sat_delta[:, 0], sat_delta[:, 1], args.levels,
                                                        meter_per_pixel)
        # corr_loss2 = Weakly_supervised_loss_w_GPS_error(corr_maps2, sat_delta[:, 0], sat_delta[:, 1], args.levels,
        #                                                 meter_per_pixel)
        # corr_loss1=0
        # corr_loss2=0

        # consistency_loss = consistency_constraint_KL_divergency(corr_maps1, corr_maps2, args.levels)

        # loss = corr_loss + args.GPS_error_coe * GPS_loss
        # loss = args.corr_weight * (corr_loss1 + corr_loss2) + args.consitency_weight * consistency_loss

        loss = corr_loss1

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            # 'corr_loss': f'{corr_loss1.item() + corr_loss2.item():.4f}',
            'corr_loss': f'{corr_loss1.item():.4f}',
            # 'consis_loss': f'{consistency_loss.item():.4f}',
            'batch_loss': loss.item(),
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'corr1_loss': corr_loss1,
                       # 'corr2_loss': corr_loss2,
                       # 'consistency_loss': consistency_loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       'batch_loss': loss, })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        if i_batch % 25 == 0 and args.visualize:
            save_path1 = f"./vis/{args.model_name}/train/{epoch}/corr1/{sat_gps[0].cpu().numpy()}.png"
            vis_corr(corr_maps1[2][0][0], sat[0], pano1[0], gt_points[0], None, save_path1)
            # save_path2 = f"./vis/{args.model_name}/train/{epoch}/corr2/{sat_gps[0].cpu().numpy()}.png"
            # vis_corr(corr_maps2[2][0][0], sat[0], pano2[0], gt_points[0], None, save_path2)

    # 返回平均损失
    return total_loss / len(train_loader)


def validate(args, model, val_loader, criterion, device, epoch=-1, vis=False):
    model.eval()
    total_loss = 0
    all_errors = []

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano = [
                x.to(device) for x in data_blob]

            # 前向传播
            sat_feat_dict, sat_conf_dict, bev_feat_dict, bev_conf_dict, mask1_dict = model(sat, resized_pano, None,
                                                                                           None, None, meter_per_pixel)

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
            if i_batch % 25 == 0 and args.visualize:
                if epoch == -1:
                    save_path = f"./vis/{args.model_name}/test/{sat_gps[0].cpu().numpy()}.png"
                else:
                    save_path = f"./vis/{args.model_name}/val/{epoch}/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)


    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)

    metrics = [1, 3, 5]
    mean_dis = np.mean(distance)
    median_dis = np.median(distance)

    if args.wandb:
        wandb.log({'val_mean': mean_dis,
                   'val_median': median_dis, })

    print(f"mean distance: {mean_dis:.4f}")
    print(f"median distance: {median_dis:.4f}")

    return mean_dis, median_dis


def train(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, val_loader = fetch_dataloader(args)

    # 模型初始化
    model = LocalizationNet(args).to(device)


            # 损失函数
    criterion = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 学习率
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度


    if args.restore_ckpt is not None:
        PATH = args.restore_ckpt  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print("Have load state_dict from: {}".format(args.restore_ckpt))


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    # 训练循环
    best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # mean_error, median_error = validate(args, model, val_loader, criterion, device, epoch)

        # 训练
        train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device, epoch)

        # 验证
        mean_error, median_error = validate(args, model, val_loader, criterion, device, epoch)

        # 学习率调度
        # scheduler.step()

        # 模型检查点
        if mean_error < best_val_mean_err:
            best_val_mean_err = mean_error

            model_path = args.save_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Directory created: {model_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')
            print(f"Saved new best model with mean error: {best_val_mean_err:.4f}")

        # 打印训练总结
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean Error: {mean_error:.4f}")
        print(f"Median Error: {median_error:.4f}")

    return model


def test(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    model = LocalizationNet(args).to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    criterion = nn.CrossEntropyLoss()
    mean_error, median_error = validate(args, model, test_loader, criterion, device)

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
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="cross-adam-360fov-infer", help="none")
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
        wandb.init(project="proj_feat", name=args.name, config=config)

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
        train(config)
    else:
        # pass
        test(config)
        # config['model'] = "/data/test/code/multi-local/location_model/cross-proj-feat-l1consistency-corr_w50_20241221_194330.pth"
        # test(config)

    if config['wandb']:
        wandb.finish()
