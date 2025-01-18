import argparse
import copy
import json
import os

import cv2
import numpy as np
from matplotlib.colors import Normalize

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['WANDB_MODE'] = "offline"
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
    consistency_constraint_KL_divergency, cross_entropy, generate_gaussian_heatmap
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


def train_epoch_distillation(args, teacher, student, train_loader, criterion, optimizer, device, epoch):
    student.train()
    teacher.eval()
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
        s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, \
            s_g2s_conf_dict, s_mask1_dict = student(sat, pano1, ones1, None, None, meter_per_pixel)

        student_corr = student.calc_corr_for_train(s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, s_g2s_conf_dict, s_mask1_dict)

        t_sat_feat_dict, t_sat_conf_dict, t_g2s_feat_dict, \
            t_g2s_conf_dict, t_mask1_dict = teacher(sat, resized_pano, None, None, None,
                                                                           meter_per_pixel)

        teacher_corr = teacher.calc_corr_for_train(t_sat_feat_dict, t_sat_conf_dict, t_g2s_feat_dict, t_g2s_conf_dict,
                                                   None)
        teacher_corr = generate_gaussian_heatmap(teacher_corr, args.levels, args.sigma)
        loss = cross_entropy(student_corr, teacher_corr, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)

        # 计算损失
        # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        # corr_loss1 = Weakly_supervised_loss_w_GPS_error(corr_maps1, sat_delta[:, 0], sat_delta[:, 1], args.levels,
        #                                                 meter_per_pixel)
        # corr_loss2 = Weakly_supervised_loss_w_GPS_error(corr_maps2, sat_delta[:, 0], sat_delta[:, 1], args.levels,
        #                                                 meter_per_pixel)
        # corr_loss1=0
        # corr_loss2=0

        # consistency_loss = consistency_constraint_KL_divergency(corr_maps1, corr_maps2, args.levels)

        # loss = corr_loss + args.GPS_error_coe * GPS_loss
        # loss = args.corr_weight * (corr_loss1 + corr_loss2) + args.consitency_weight * consistency_loss

        # loss = corr_loss1

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            # 'corr_loss': f'{corr_loss1.item() + corr_loss2.item():.4f}',
            'ce_loss': f'{loss.item():.4f}',
            # 'consis_loss': f'{consistency_loss.item():.4f}',
            # 'batch_loss': loss.item(),
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       # 'corr2_loss': corr_loss2,
                       # 'consistency_loss': consistency_loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        if i_batch % 25 == 0 and args.visualize:
            if args.save_visualization:
                save_path1 = f"./vis/distillation/{args.model_name}/train/{epoch}/student/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(student_corr[2][0], sat[0], pano1[0], gt_points[0], None, save_path1, temp=args.student_temp)
                save_path2 = f"./vis/distillation/{args.model_name}/train/{epoch}/teacher/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(teacher_corr[2][0], sat[0], resized_pano[0], gt_points[0], None, save_path2, temp=args.teacher_temp)
            else:
                vis_corr(student_corr[2][0], sat[0], pano1[0], gt_points[0], None, None, temp=args.student_temp)
                vis_corr(teacher_corr[2][0], sat[0], resized_pano[0], gt_points[0], None, None, temp=args.teacher_temp)

    # 返回平均损失
    return total_loss / len(train_loader)


def train_epoch_self_distillation(args, model, train_loader, criterion, optimizer, device, epoch):
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
        sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, g2s2_feat_dict,\
            g2s2_conf_dict, mask1_dict, mask2_dict = model(sat, resized_pano, ones1, pano1, ones1, meter_per_pixel)

        corr_maps1 = model.calc_corr_for_train(sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict,
                                               mask_dict=None, batch_wise=True)
        corr_maps2 = model.calc_corr_for_train(sat_feat_dict, sat_conf_dict, g2s2_feat_dict, g2s2_conf_dict,
                                               mask_dict=mask2_dict, batch_wise=False)

        # loss = cross_entropy(student_corr, teacher_corr, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)

        # 计算损失
        # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        corr_loss1 = Weakly_supervised_loss_w_GPS_error(corr_maps1, sat_delta[:, 0], sat_delta[:, 1], args.levels,
                                                        meter_per_pixel)
        corr1_pos = {}
        for _, level in enumerate(args.levels):
            corr = corr_maps1[level]
            B, _, _, _ = corr.shape
            pos_corr1 = corr[torch.arange(B), torch.arange(B)]
            corr1_pos[level] = pos_corr1

        ce_loss = cross_entropy(corr_maps2, corr1_pos, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)
        # corr_loss2 = Weakly_supervised_loss_w_GPS_error(corr_maps2, sat_delta[:, 0], sat_delta[:, 1], args.levels,
        #                                                 meter_per_pixel)
        # corr_loss1=0
        # corr_loss2=0

        # consistency_loss = consistency_constraint_KL_divergency(corr_maps1, corr_maps2, args.levels)

        # loss = corr_loss + args.GPS_error_coe * GPS_loss
        # loss = args.corr_weight * (corr_loss1 + corr_loss2) + args.consitency_weight * consistency_loss

        loss = corr_loss1 + args.self_distillation_w * ce_loss

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
            'corr_loss': f'{corr_loss1.item():.4f}',
            'ce_loss': f'{ce_loss.item():.4f}',
            # 'consis_loss': f'{consistency_loss.item():.4f}',
            # 'batch_loss': loss.item(),
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': ce_loss,
                       'corr1_loss': corr_loss1,
                       # 'consistency_loss': consistency_loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        if i_batch % 25 == 0 and args.visualize:
            if args.save_visualization:
                save_path1 = f"./vis/distillation/{args.model_name}/train/{epoch}/ori/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(corr1_pos[2][0], sat[0], resized_pano[0], gt_points[0], None, save_path1, temp=1)
                save_path2 = f"./vis/distillation/{args.model_name}/train/{epoch}/student/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(corr_maps2[2][0], sat[0], pano1[0], gt_points[0], None, save_path2, temp=1)
            else:
                vis_corr(corr1_pos[2][0], sat[0], pano1[0], gt_points[0], None, None, temp=args.student_temp)

    # 返回平均损失
    return total_loss / len(train_loader)



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
        s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, \
            s_g2s_conf_dict, s_mask1_dict = model(sat, pano1, ones1, None, None, meter_per_pixel)

        corr_maps1 = model.calc_corr_for_train(s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, s_g2s_conf_dict, s_mask1_dict)


        # loss = cross_entropy(student_corr, teacher_corr, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)

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
            'ce_loss': f'{loss.item():.4f}',
            # 'consis_loss': f'{consistency_loss.item():.4f}',
            # 'batch_loss': loss.item(),
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       # 'corr2_loss': corr_loss2,
                       # 'consistency_loss': consistency_loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        if i_batch % 25 == 0 and args.visualize:
            if args.save_visualization:
                save_path1 = f"./vis/distillation/{args.model_name}/train/{epoch}/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(corr_maps1[2][0], sat[0], pano1[0], gt_points[0], None, save_path1, temp=args.student_temp)
            else:
                vis_corr(corr_maps1[2][0], sat[0], pano1[0], gt_points[0], None, None, temp=args.student_temp)

    # 返回平均损失
    return total_loss / len(train_loader)



def validate(args, model, val_loader, criterion, device, epoch=-1, vis=False, name=None):
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
                if args.save_visualization:
                    if epoch == -1:
                        save_path = f"./vis/distillation/{args.model_name}/test/{sat_gps[0].cpu().numpy()}.png"
                    else:
                        if name is None:
                            save_path = f"./vis/distillation/{args.model_name}/val/{epoch}/{sat_gps[0].cpu().numpy()}.png"
                        else:
                            save_path = f"./vis/distillation/{args.model_name}/val/{name}_{epoch}/{sat_gps[0].cpu().numpy()}.png"
                    vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)
                else:
                    vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], None)


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
        if name is not None:
            wandb.log({f'val_{name}_mean': mean_dis,
                       f'val_{name}_median': median_dis, })
        else:
            wandb.log({'val_mean': mean_dis,
                   'val_median': median_dis, })


    print(f"mean distance: {mean_dis:.4f}")
    print(f"median distance: {median_dis:.4f}")

    return mean_dis, median_dis


def validate_distillation(args, teacher, student, val_loader, criterion, device, epoch=-1, vis=False, name=None):
    teacher.eval()
    student.eval()
    total_loss = 0
    all_errors = []

    pred_us_t = []
    pred_vs_t = []

    gt_us = []
    gt_vs = []

    pred_us_s = []
    pred_vs_s = []

    pred_us_max = []
    pred_vs_max = []


    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta, meter_per_pixel, pano1, ones1, pano2, ones2, resized_pano = [
                x.to(device) for x in data_blob]

            # 前向传播
            sat_feat_dict_t, sat_conf_dict_t, bev_feat_dict_t, bev_conf_dict_t, mask1_dict = teacher(sat, resized_pano, None,
                                                                                           None, None, meter_per_pixel)

            corr_t = teacher.calc_corr_for_val(sat_feat_dict_t, sat_conf_dict_t, bev_feat_dict_t, bev_conf_dict_t, None)

            sat_feat_dict_s, sat_conf_dict_s, bev_feat_dict_s, bev_conf_dict_s, mask1_dict = student(sat, resized_pano, None,
                                                                                           None, None, meter_per_pixel)

            corr_s = student.calc_corr_for_val(sat_feat_dict_s, sat_conf_dict_s, bev_feat_dict_s, bev_conf_dict_s, None)

            # # 计算损失
            # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
            # loss = 100 * cls_loss + 1 * reg_loss

            max_level = args.levels[-1]

            B, corr_H, corr_W = corr_t.shape

            # teacher or before
            max_value_t, max_index_t = torch.max(corr_t.reshape(B, -1), dim=1)
            pred_u_t = (max_index_t % corr_W - corr_W / 2)
            pred_v_t = (max_index_t // corr_W - corr_H / 2)

            pred_u_t = pred_u_t * np.power(2, 3 - max_level) * meter_per_pixel
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
            max_us_temp = []
            max_vs_temp = []

            for i in range(max_value_t.shape[0]):  # 遍历 batch
                if max_value_t[i] > max_value_s[i]:
                    max_us_temp.append(pred_u_t[i].data.cpu())  # 保证维度一致
                    max_vs_temp.append(pred_v_t[i].data.cpu())
                else:
                    max_us_temp.append(pred_u_s[i].data.cpu())
                    max_vs_temp.append(pred_v_s[i].data.cpu())
            max_us_temp = np.array([t.numpy() for t in max_us_temp])  # 直接转换为 NumPy 数组
            max_vs_temp = np.array([t.numpy() for t in max_vs_temp])  # 同上
            pred_us_max.append(max_us_temp)
            pred_vs_max.append(max_vs_temp)
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

            # pred_us_t_ = np.concatenate(pred_us_t, axis=0)
            # pred_vs_t_ = np.concatenate(pred_vs_t, axis=0)
            #
            # pred_us_s_ = np.concatenate(pred_us_s, axis=0)
            # pred_vs_s_ = np.concatenate(pred_vs_s, axis=0)



            distance_t = np.sqrt((pred_u_t.cpu().numpy() - gt_shift_u.data.cpu().numpy()) ** 2 + (pred_v_t.cpu().numpy() - gt_shift_v.data.cpu().numpy()) ** 2)  # [N]
            distance_s = np.sqrt((pred_u_s.cpu().numpy() - gt_shift_u.data.cpu().numpy()) ** 2 + (pred_v_s.cpu().numpy() - gt_shift_v.data.cpu().numpy()) ** 2)  # [N]

            s_worse = np.where((distance_s > distance_t) & ((distance_s - distance_t) > 2))[0]

            if len(s_worse) > 0 and args.visualize:

                for i in s_worse:
                    save_path = None
                    if args.save_visualization:
                        save_path = f"./vis/distillation/{args.model_name}/s_worse/{sat_gps[i].cpu().numpy()}.jpg"
                    vis_two_sat(corr_t[i], corr_s[i], sat[i], resized_pano[i], gt_points[i], [pred_x_t[i], pred_y_t[i]], [pred_x_s[i], pred_y_s[i]], save_path)

            t_worse = np.where((distance_s < distance_t) & ((distance_t - distance_s) > 2))[0]
            if len(t_worse) > 0 and args.visualize:

                for i in t_worse:
                    save_path = None
                    if args.save_visualization:
                        save_path = f"./vis/distillation/{args.model_name}/t_worse/{sat_gps[i].cpu().numpy()}.jpg"
                    vis_two_sat(corr_t[i], corr_s[i], sat[i], resized_pano[i], gt_points[i], [pred_x_t[i], pred_y_t[i]],
                                [pred_x_s[i], pred_y_s[i]], save_path)

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

    pred_us_max = np.concatenate(pred_us_max, axis=0)
    pred_vs_max = np.concatenate(pred_vs_max, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    distance_t = np.sqrt((pred_us_t - gt_us) ** 2 + (pred_vs_t - gt_vs) ** 2)  # [N]
    distance_s = np.sqrt((pred_us_s - gt_us) ** 2 + (pred_vs_s - gt_vs) ** 2)  # [N]
    distance_max = np.sqrt((pred_us_max - gt_us) ** 2 + (pred_vs_max - gt_vs) ** 2)
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)
    # np.save('distance_t.npy', distance_t)
    # np.save('distance_s.npy', distance_s)

    visualize_distributions(distance_t, distance_s, f"./vis/distillation/{args.model_name}")



    metrics = [1, 3, 5]
    mean_dis_t = np.mean(distance_t)
    median_dis_t = np.median(distance_t)

    mean_dis_s = np.mean(distance_s)
    median_dis_s = np.median(distance_s)

    mean_dis_max = np.mean(distance_max)
    median_dis_max = np.median(distance_max)

    if args.wandb:
        wandb.log({'val_before_mean': mean_dis_t,
                   'val_before_median': median_dis_t,
                   'val_student_mean': mean_dis_s,
                   'val_student_median': median_dis_s,
                   'val_maxP_mean': mean_dis_max,
                   'val_maxP_median': median_dis_max,
                   })


    print(f"before distillation mean distance: {mean_dis_t:.4f}")
    print(f"before distillation distance: {median_dis_t:.4f}")
    print(f"after distillation mean distance: {mean_dis_s:.4f}")
    print(f"after distillation distance: {median_dis_s:.4f}")
    print(f"max p mean distance: {mean_dis_max:.4f}")
    print(f"max p median distance: {median_dis_max:.4f}")

    return mean_dis_t, mean_dis_s, median_dis_t, median_dis_s



def test_(args, model, val_loader, criterion, device, epoch=-1, vis=False, name=None):
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
            sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, g2s2_feat_dict,\
                g2s2_conf_dict, mask1_dict, mask2_dict = model(sat, pano1, ones1, pano2, ones2, meter_per_pixel)

            corr1 = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, g2s1_feat_dict, g2s1_conf_dict, mask1_dict)
            corr2 = model.calc_corr_for_val(sat_feat_dict, sat_conf_dict, g2s2_feat_dict, g2s2_conf_dict, mask2_dict)

            # # 计算损失
            # cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
            # loss = 100 * cls_loss + 1 * reg_loss

            max_level = args.levels[-1]
            corr = corr2 + corr1

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
                    vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], save_path)
                else:
                    vis_corr(corr[0], sat[0], resized_pano[0], gt_points[0], [pred_x[0], pred_y[0]], None)


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



def train_distillation(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    # 数据加载
    vigor = VIGOR(args, "train")

    train_loader, val_loader = fetch_dataloader(args, vigor)

    # 模型初始化
    student = LocalizationNet(args).to(device)


            # 损失函数
    criterion = None

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-4,  # 学习率
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度


    if args.restore_ckpt is not None:
        PATH = args.restore_ckpt  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            student.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(args.restore_ckpt))


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False



    # 训练循环
    t_best_val_mean_err = float('inf')
    s_best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        vigor.update_fov()
        # mean_error, median_error = validate(args, model, val_loader, criterion, device, epoch)

        # 训练
        train_loss = train_epoch_distillation(args, teacher, student, train_loader, criterion, optimizer, device, epoch)

        update_teacher_model(student, teacher, args.ema)

        # 验证

        s_mean_error, s_median_error = validate(args, student, val_loader, criterion, device, epoch, name="student")

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

        t_mean_error, t_median_error = validate(args, teacher, val_loader, criterion, device, epoch, name="teacher")

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


def train(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    # 数据加载
    vigor = VIGOR(args, "train")

    train_loader, val_loader = fetch_dataloader(args, vigor)

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
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['lr_schedule'])
            print_colored("Have load state_dict from: {}".format(args.restore_ckpt))


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )



    # 训练循环
    s_best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # mean_error, median_error = validate(args, model, val_loader, criterion, device, epoch)

        # 训练
        # train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device, epoch)
        train_loss = train_epoch_self_distillation(args, model, train_loader, criterion, optimizer, device, epoch)

        # 验证

        s_mean_error, s_median_error = validate(args, model, val_loader, criterion, device, epoch, name="student")

        if s_mean_error < s_best_val_mean_err:
            s_best_val_mean_err = s_mean_error

            model_path = os.path.join(args.save_path, "student")
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


def vis_distillation_err(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    teacher = LocalizationNet(args).to(device)

    teacher, start_epoch, best_val_loss = load_trained_model(teacher, args.teacher, device)

    student = LocalizationNet(args).to(device)

    student, start_epoch, best_val_loss = load_trained_model(student, args.student, device)
    criterion = nn.CrossEntropyLoss()
    validate_distillation(args, teacher, student, test_loader, criterion, device)

    # print(f"Val Loss: {val_loss:.4f}")
    # print(f"mean rotation: {mean_error:.4f}")
    # print(f"median rotation: {median_error:.4f}")

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

    parser.add_argument('--name', default="cross-fov360-240-w1", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=True)

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
        wandb.init(project="self-distillation", name=args.name, config=config)

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
        # train_distillation(config)
        train(config)
    else:
        # pass
        test(config)
        # vis_distillation_err(config)
        # config['model'] = "/data/test/code/multi-local/location_model/cross-proj-feat-l1consistency-corr_w50_20241221_194330.pth"
        # test(config)

    if config['wandb']:
        wandb.finish()
