import argparse
import copy
import json
import os
import random

import numpy as np

from model.dino import center_padding, DINO
from model.loss import cross_entropy, multi_scale_contrastive_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['WANDB_MODE'] = "offline"
import time

import torch
import wandb
from easydict import EasyDict
from torch import nn, optim

from tqdm import tqdm

from model.network_vigor import LocalizationNet

from utils.util import setup_seed, print_colored, vis_corr, generate_mask_avg, generate_MAE_mask
from dataset.VIGOR import fetch_dataloader, VIGOR



def load_trained_model(model, pth_file, device):
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
        # 解包数据并移动到设备
        bev, sat, pano_gps, sat_gps, ori_angle, sat_delta, meter_per_pixel, masked_pano, mask, resized_pano, rotated_pano, city, masked_fov = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
        city = data_blob[-1]

        optimizer.zero_grad()

        sat_img = 2 * (sat / 255.0) - 1.0
        pano_img = 2 * (resized_pano / 255.0) - 1.0

        sat_img = sat_img.contiguous()
        pano_img = pano_img.contiguous()

        pano_img = pano_img.permute(0, 3, 1, 2)

        sat_feat_list = dino(sat_img)
        pano_feat_list = dino(pano_img)

        copied_sat_feat_list = [t.clone() for t in sat_feat_list]

        # forward
        (t_sat_feat_dict, t_sat_conf_dict, t_g2s_feat_dict, t_g2s_conf_dict, t_mask_dict, t_pano_conf_dict,
         t_pano1_feat_dict) = teacher(sat_feat_list, pano_feat_list, meter_per_pixel, mask=None)

        # different masking strategy
        mask_pano = mask
        if args.mask_fov:
            pass
        elif args.mask_activation:
            mask_pano = generate_mask_avg(t_pano1_feat_dict[3], args.mask_ratio)
            mask_pano = mask_pano.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
            masked_pano = resized_pano * mask_pano
        elif args.mask_MAE:
            resized_pano = center_padding(resized_pano.permute(0, 3, 1, 2), 14)
            resized_pano = resized_pano.permute(0, 2, 3, 1)
            masked_pano = resized_pano
            mask_pano = torch.ones_like(resized_pano)[:, :, :, :3]
            imgs = resized_pano.cpu().numpy()
            for i in range(resized_pano.shape[0]):
                r = random.uniform(120/360, 180/360)
                mask = generate_MAE_mask(imgs[i], r, patch_size=14)
                mask = torch.from_numpy(mask).to(resized_pano.device)
                mask = mask.unsqueeze(-1).repeat(1, 1, 3)
                mask_pano[i] = mask
                masked_pano[i] = resized_pano[i] * mask

        masked_pano_img = 2 * (masked_pano / 255.0) - 1.0
        masked_pano_img = masked_pano_img.contiguous()
        masked_pano_img = masked_pano_img.permute(0, 3, 1, 2)


        masked_pano_feat_list = dino(masked_pano_img)

        s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, \
            s_g2s_conf_dict, s_mask_dict, s_pano_conf_dict, s_pano_feat_dict = student(copied_sat_feat_list, masked_pano_feat_list, meter_per_pixel, mask=mask_pano)

        student_corr = student.calc_corr_for_train(s_sat_feat_dict, s_g2s_feat_dict, mask_dict=s_mask_dict)
        teacher_corr = teacher.calc_corr_for_train(t_sat_feat_dict, t_g2s_feat_dict, mask_dict=None)


        loss = cross_entropy(student_corr, teacher_corr, args.levels, s_temp=args.student_temp, t_temp=args.teacher_temp)


        # bp
        loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        # student param update
        optimizer.step()

        # loss cumulation
        total_loss += loss.item()

        # update pbar
        pbar.set_postfix({
            'ce_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        # visualization
        if i_batch % args.vis_freq == 0 and args.visualize:
            if args.save_visualization:
                save_path10 = f"./vis/distillation/{args.model_name}/train/{epoch}/student0/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(student_corr[2][0], sat[0], masked_pano[0], gt_points[0], None, save_path10, temp=args.student_temp)

                save_path2 = f"./vis/distillation/{args.model_name}/train/{epoch}/teacher/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(teacher_corr[2][0], sat[0], resized_pano[0], gt_points[0], None, save_path2, temp=args.teacher_temp)
            else:
                vis_corr(student_corr[2][0], sat[0], masked_pano[0], gt_points[0], None, None, temp=args.student_temp)
                vis_corr(teacher_corr[2][0], sat[0], resized_pano[0], gt_points[0], None, None, temp=args.teacher_temp)

    return total_loss / len(train_loader)


def train_epoch_g2sweakly(args, dino, model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    # 进度条
    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        # 解包数据并移动到设备
        bev, sat, pano_gps, sat_gps, ori_angle, sat_delta, meter_per_pixel, masked_pano, mask, resized_pano, rotated_pano, city, masked_fov = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
        city = data_blob[-1]


        # 清除梯度
        optimizer.zero_grad()

        sat_img = 2 * (sat / 255.0) - 1.0
        pano_img = 2 * (resized_pano / 255.0) - 1.0

        sat_img = sat_img.contiguous()
        pano_img = pano_img.contiguous()

        pano_img = pano_img.permute(0, 3, 1, 2)

        sat_feat_list = dino(sat_img)
        pano_feat_list = dino(pano_img)


        # 前向传播
        (s_sat_feat_dict, s_sat_conf_dict, s_g2s_feat_dict, s_g2s_conf_dict, s_mask1_dict
         , pano1_conf_dict, pano1_feat_dict) = model(sat_feat_list, pano_feat_list, meter_per_pixel, mask=None)

        corr_maps = model.calc_corr_for_train(s_sat_feat_dict, s_g2s_feat_dict, mask_dict=None, batch_wise=True)

        loss = multi_scale_contrastive_loss(corr_maps, args.levels)


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            'ce_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'ce_loss': loss,
                       'avg_loss': total_loss / (i_batch + 1),
                       })

        gt_points = sat_delta * 512 / 4
        gt_points[:, 0] = 512 / 2 + gt_points[:, 0]
        gt_points[:, 1] = 512 / 2 + gt_points[:, 1]

        if i_batch % args.vis_freq == 0 and args.visualize:
            if args.save_visualization:
                save_path1 = f"./vis/distillation/{args.model_name}/train/{epoch}/{sat_gps[0].cpu().numpy()}.png"
                vis_corr(corr_maps[2][0][0], sat[0], resized_pano[0], gt_points[0], None, save_path1, temp=args.student_temp)
            else:
                vis_corr(corr_maps[2][0][0], sat[0], resized_pano[0], gt_points[0], None, None, temp=args.student_temp)

    return total_loss / len(train_loader)



def validate(args, dino, model, val_loader, device, epoch=-1, vis=False, name=None):
    model.eval()
    all_errors = []

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            bev, sat, pano_gps, sat_gps, ori_angle, sat_delta, meter_per_pixel, masked_pano, mask, resized_pano, rotated_pano, city, masked_fov = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
            city = data_blob[-1]

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


            max_level = args.levels[-1]

            B, corr_H, corr_W = corr.shape

            max_index = torch.argmax(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2)
            pred_v = (max_index // corr_W - corr_H / 2)

            _, _, feat_H, feat_W = sat_feat_dict[max_level].shape

            # pred_u = pred_u * np.power(2, 3 - max_level) * meter_per_pixel
            # pred_v = pred_v * np.power(2, 3 - max_level) * meter_per_pixel

            pred_u = pred_u * 512/feat_H * meter_per_pixel
            pred_v = pred_v * 512/feat_H * meter_per_pixel

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
            if i_batch % args.vis_freq == 0 and args.visualize:
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
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    vigor = VIGOR(args, "train")

    train_loader, val_loader = fetch_dataloader(args, vigor)

    student = LocalizationNet(args).to(device)

    dinov2 = DINO(model_name="vitb14").to(device)


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    t_best_val_mean_err = float('inf')
    s_best_val_mean_err = float('inf')

    if args.ckpt_geodistill is not None:
        PATH = args.ckpt_geodistill  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            student.load_state_dict(checkpoint['model_state_dict'])
            print_colored("Have load state_dict from: {}".format(PATH))
    elif args.student_ckpt is not None:
        PATH = args.student_ckpt  # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            s_best_val_mean_err = checkpoint['loss']
            student.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

            print_colored("Have load state_dict from: {}".format(PATH))

    for param in teacher.parameters():
        param.requires_grad = False

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch_geodistill(args, dinov2, teacher, student, train_loader, optimizer, device, epoch)

        update_teacher_model(student, teacher, args.ema)

        s_mean_error, s_median_error = validate(args, dinov2, student, val_loader, device, epoch, name="student")

        if s_mean_error < s_best_val_mean_err:
            s_best_val_mean_err = s_mean_error

            model_path = os.path.join(args.save_path, "geodistill", "student")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': s_best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best student model with mean error: {s_best_val_mean_err:.4f}")

        t_mean_error, t_median_error = validate(args, dinov2, teacher, val_loader, device, epoch, name="teacher")

        scheduler.step()

        if t_mean_error < t_best_val_mean_err:
            t_best_val_mean_err = t_mean_error

            model_path = os.path.join(args.save_path, "geodistill", "teacher")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': t_best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best teacher model with mean error: {t_best_val_mean_err:.4f}")

        area = "cross area" if args.cross_area else "same area"

        print(f"============GeoDistill {area}  Epoch {epoch + 1} Summary==============")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"teacher Mean Error: {t_mean_error:.4f}")
        print(f"teacher Median Error: {t_median_error:.4f}")
        print(f"student Mean Error: {s_mean_error:.4f}")
        print(f"student Median Error: {s_median_error:.4f}")

    return teacher, student


def train_g2sweakly(args):
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")

    vigor = VIGOR(args, "train")

    train_loader, val_loader = fetch_dataloader(args, vigor)

    model = LocalizationNet(args).to(device)

    dinov2 = DINO(model_name="vitb14").to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


    if args.ckpt_g2sweakly is not None:
        PATH = args.ckpt_g2sweakly
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



    best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # mean_error, median_error = validate(args, dinov2, model, val_loader, device, epoch, name="student")


        train_loss = train_epoch_g2sweakly(args, dinov2, model, train_loader, optimizer, device, epoch)


        mean_error, median_error = validate(args, dinov2, model, val_loader, device, epoch, name="student")

        if mean_error < best_val_mean_err:
            best_val_mean_err = mean_error

            model_path = os.path.join(args.save_path, "g2sweakly")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'loss': best_val_mean_err
            }, f'{model_path}/{args.model_name}.pth')

            print_colored(f"Saved new best model with mean error: {best_val_mean_err:.4f}")


        scheduler.step()
        area = "cross area" if args.cross_area else "same area"

        print(f"============G2SWeakly {area}  Epoch {epoch + 1} Summary==============")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"model Mean Error: {mean_error:.4f}")
        print(f"model Median Error: {median_error:.4f}")

    return model


def test(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    test_loader = fetch_dataloader(args, split="test")

    model = LocalizationNet(args).to(device)
    dinov2 = DINO(model_name="vitb14").to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    mean_error, median_error = validate(args, dinov2, model, test_loader, device, name="student")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config_vigor.json", type=str, help="path of config file")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 16, 4])

    parser.add_argument('--name', default="same-g2sweakly-clean-infer", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=False, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=False)
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
            config['epochs'] *= 2
            train_geodistill(config)
    else:
        test(config)

    if config['wandb']:
        wandb.finish()
