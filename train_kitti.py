import argparse
import copy
import json
import os
import random

import cv2
import numpy as np
from matplotlib.colors import Normalize

from dataset.KITTI import load_train_data, load_test1_data, load_test2_data
from model.dino import DINO
from model.loss import cross_entropy, multi_scale_contrastive_loss

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

from model.network_kitti_dino import LocalizationNet, get_meter_per_pixel
# from model.Network_KITTI import LocalizationNet, get_meter_per_pixel

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


def update_teacher_model(student_model, teacher_model, ema_decay=0.99):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(ema_decay).add_((1 - ema_decay) * student_param.data)


def train_epoch_geodistill(args, dino, teacher, student, train_loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    total_loss = 0

    # 进度条
    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, mask = [item.to(device) for
                                                                                                    item in data_blob]
        masked_grd = grd_left_imgs * mask
        # fig = plt.figure(figsize=(20, 5), dpi=100)  # 可自定义尺寸
        # ax = fig.add_axes([0, 0, 1, 1])  # 完全填充，没有边框
        # ax.axis('off')  # 不显示坐标轴
        # mask = masked_grd[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
        # ax.imshow(mask.astype(np.uint8))
        # plt.show()

        sat_feat_list = dino(sat_map)
        grd_feat_list = dino(grd_left_imgs)

        copied_sat_feat_list = [t.clone() for t in sat_feat_list]

        sat_feat_dict_t, g2s_feat_dict_t, mask_dict_t = teacher(sat_feat_list, grd_feat_list, left_camera_k)

        masked_grd_feat_list = dino(masked_grd)

        sat_feat_dict_s, g2s_feat_dict_s, mask_dict_s = student(copied_sat_feat_list, masked_grd_feat_list, left_camera_k, mask)

        # 清除梯度
        optimizer.zero_grad()

        teacher_corr = teacher.calc_corr_for_train(sat_feat_dict_t, g2s_feat_dict_t, batch_wise=False)
        student_corr = teacher.calc_corr_for_train(sat_feat_dict_s, g2s_feat_dict_s, batch_wise=False, mask_dict=mask_dict_s)


        loss = cross_entropy(student_corr, teacher_corr, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 更新进度条

        # 更新进度条
        pbar.set_postfix({
            'ce_loss': f'{loss.item():.4f}',

            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })


    return total_loss / len(train_loader)



def train_epoch_g2sweakly(args, dino, model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    # 进度条
    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        # 解包数据并移动到设备
        sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for
                                                                                                    item in data_blob[:7]]

        sat_feat_list = dino(sat_map)
        grd_feat_list = dino(grd_left_imgs)

        # sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, fov_mask_dict = \
        #     model(sat_map, grd_left_imgs, left_camera_k)
        sat_feat_dict, g2s_feat_dict, mask_dict = model(sat_feat_list, grd_feat_list, left_camera_k)

        # 清除梯度
        optimizer.zero_grad()

        corr_maps = model.calc_corr_for_train(sat_feat_dict, g2s_feat_dict, batch_wise=True)

        loss = multi_scale_contrastive_loss(corr_maps, args.levels)


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
            'ce_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

    return total_loss / len(train_loader)



def validate(args, dino, model, val_loader, device, epoch=-1, vis=False, name=None):
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
            # sat_align_cam和sat_map都是256*256
            sat_align_cam, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [
                item.to(device) for item in data_blob[:7]]

            sat_feat_list = dino(sat_map)
            grd_feat_list = dino(grd_left_imgs)

            # plt.figure(figsize=(10, 5))
            # plt.imshow((sat_map[0]*256).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
            # plt.show()
            # 原版的网络输出最大的feature是256*256 和输入图像一样大
            sat_feat_dict, g2s_feat_dict, mask_dict = model(sat_feat_list, grd_feat_list, left_camera_k)

            meter_per_pixel = get_meter_per_pixel(scale=sat_feat_dict[2].shape[-1]/sat_map.shape[-1])

            corr = model.calc_corr_for_val(sat_feat_dict, g2s_feat_dict, mask_dict=None)

            max_level = args.levels[-1]

            B, corr_H, corr_W = corr.shape

            max_index = torch.argmax(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)

            pred_u = pred_u * cos + pred_v * sin
            pred_v = - pred_u * sin + pred_v * cos


            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_us.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_vs.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)


    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]

    metrics = [1, 3, 5]
    mean_dis = np.mean(distance)
    median_dis = np.median(distance)

    if args.wandb:
        if name is not None:
            wandb.log({f'val_{name}_mean': mean_dis,
                       f'val_{name}_median': median_dis, })
        else:
            wandb.log({'val_mean': mean_dis,
                   'val_median': median_dis, })


    print(f"mean distance: {mean_dis:.4f}")
    print(f"median distance: {median_dis:.4f}")

    return mean_dis, median_dis



def train_geodistill(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    # 数据加载

    train_loader = load_train_data(args, args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    val_loader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

    # 模型初始化
    student = LocalizationNet(args).to(device)
    dinov2 = DINO().to(device)


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-4,  # 学习率
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度
    t_best_val_mean_err = float('inf')
    s_best_val_mean_err = float('inf')

    if args.ckpt_geodistill is not None:
        PATH = args.ckpt_geodistill  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            student.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(PATH))
    elif args.student_ckpt is not None:
        PATH = args.student_ckpt  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            s_best_val_mean_err = checkpoint['loss']
            student.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(PATH))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    teacher = copy.deepcopy(student)
    if args.teacher_ckpt is not None:
        PATH = args.teacher_ckpt
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            t_best_val_mean_err = checkpoint['loss']
            teacher.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(args.restore_ckpt))

    for param in teacher.parameters():
        param.requires_grad = False

    # 训练循环

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch_geodistill(args, dinov2, teacher, student, train_loader, optimizer, device, epoch)

        update_teacher_model(student, teacher, args.ema)

        # 验证

        s_mean_error, s_median_error = validate(args, dinov2, student, val_loader, device, epoch, name="student")

        if s_mean_error < s_best_val_mean_err:
            s_best_val_mean_err = s_mean_error

            model_path = os.path.join(args.save_path, "student")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Directory created: {model_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': s_best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best student model with mean error: {s_best_val_mean_err:.4f}")

        t_mean_error, t_median_error = validate(args, dinov2, teacher, val_loader, device, epoch, name="teacher")

        # 学习率调度
        scheduler.step()

        # 模型检查点
        if t_mean_error < t_best_val_mean_err:
            t_best_val_mean_err = t_mean_error

            model_path = os.path.join(args.save_path, "teacher")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Directory created: {model_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': t_best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best teacher model with mean error: {t_best_val_mean_err:.4f}")

        # 打印训练总结
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_loss:.4f}")
        print(f"teacher Mean Error: {t_mean_error:.4f}")
        print(f"teacher Median Error: {t_median_error:.4f}")
        print(f"student Mean Error: {s_mean_error:.4f}")
        print(f"student Median Error: {s_median_error:.4f}")

    return teacher, student


def train_g2sweakly(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    # 数据加载

    train_loader = load_train_data(args, args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    val_loader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)


    # 模型初始化
    model = LocalizationNet(args).to(device)

    dinov2 = DINO().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 学习率
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度


    if args.ckpt_g2sweakly is not None:
        PATH = args.ckpt_g2sweakly  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(PATH))


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    s_best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # s_mean_error, s_median_error = validate(args, dinov2, model, val_loader, device, epoch, name="student")


        # training
        train_loss = train_epoch_g2sweakly(args, dinov2, model, train_loader, optimizer, device, epoch)

        # validation
        s_mean_error, s_median_error = validate(args, dinov2, model, val_loader, device, epoch, name="student")

        if s_mean_error < s_best_val_mean_err:
            s_best_val_mean_err = s_mean_error

            model_path = os.path.join(args.save_path, "vanilla")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Directory created: {model_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': s_best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best student model with mean error: {s_best_val_mean_err:.4f}")

        # 学习率调度
        scheduler.step()

        # 打印训练总结
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_loss:.4f}")
        print(f"model Mean Error: {s_mean_error:.4f}")
        print(f"model Median Error: {s_median_error:.4f}")

    return model


def test_cross(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

    model = LocalizationNet(args).to(device)
    dinov2 = DINO().to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    mean_error, median_error = validate(args, dinov2, model, test_loader, device, name="student")

def test_same(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

    model = LocalizationNet(args).to(device)
    dinov2 = DINO().to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    criterion = nn.CrossEntropyLoss()
    mean_error, median_error = validate(args, dinov2, model, test_loader, device, name="student")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config_kitti.json", type=str, help="path of config file")
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--rotation_range', type=float, default=0, help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')


    parser.add_argument('--name', default="kitti-dino-geodistill", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=True)

    parser.add_argument('--train_g2sweakly', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['validation'] = args.validation
    config['name'] = args.name
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    config['train'] = args.train
    config['epochs'] = args.epochs
    config['rotation_range'] = args.rotation_range
    config['shift_range_lat'] = args.shift_range_lat
    config['shift_range_lon'] = args.shift_range_lon

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

    if not os.path.isdir(config['save_path']):
        os.mkdir(config['save_path'])


    if args.train:
        if args.train_g2sweakly:
            train_g2sweakly(config)
        else:
            train_geodistill(config)
    else:
        if args.cross_area:
            print("==========test in cross area==========")
            test_cross(config)
        print("==========test in same area==========")
        test_same(config)

    if config['wandb']:
        wandb.finish()
