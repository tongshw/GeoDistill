import argparse
import json
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import time

import torch
import wandb
# import wandb
from easydict import EasyDict
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Network import RotationPredictionNet
from model.loss import RotationLoss
from utils.util import setup_seed, print_colored, count_parameters, visualization, TextColors
from dataset.VIGOR import fetch_dataloader


def load_trained_model(model, pth_file, device):
    # 加载保存的模型权重
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {pth_file}, epoch {checkpoint['epoch']}, val_loss {checkpoint['loss']:.4f}")
    return model, checkpoint['epoch'], checkpoint['loss']


def train_epoch(args, model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    # 进度条
    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        # 解包数据并移动到设备
        bev, sat, grd_gps, sat_gps, ori_angle, sat_delta = [x.to(device) for x in data_blob]


        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        # 根据你的网络结构调整输入
        pred_label = model(sat, bev)  # 或者 model(bev, sat)

        # 计算损失
        loss = criterion(pred_label, soft_rotation_label)

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
            'batch_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

    # 返回平均损失
    return total_loss / len(train_loader)


def validate(args, model, val_loader, criterion, device, vis=False):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mean = 0
    all_errors = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):

            bev, sat, grd_gps, sat_gps, ori_angle, sat_delta = [x.to(device) for x in data_blob]


            # 前向传播
            pred_label = model(sat, bev)

            err, mean_err, median_err = calculate_errors(gt_ori, pred_label)

            # 计算损失
            loss = criterion(pred_label, soft_rotation_label)

            # 计算平均绝对误差
            # mae = torch.mean(torch.abs(pred_label - rotation_label))

            # 累计损失和误差
            total_loss += loss.item()
            # total_mae += mae.item()
            total_mean += mean_err.item()

            all_errors.extend(err.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                'val_batch_loss': f'{loss.item():.4f}',
                'mean_rotation_err': f'{mean_err.item():.4f}'
            })

            if vis:
                predicted_labels = torch.argmax(pred_label, dim=1)
                visualization(bev, sat, sat_delta, ori_angle, predicted_labels - 45)

    # 计算平均指标
    avg_loss = total_loss / len(val_loader)
    overall_median = np.median(all_errors)
    overall_mean = np.mean(all_errors)

    return avg_loss, overall_mean, overall_median


def train(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))
    start_time = time.strftime('%Y%m%d_%H%M%S')

    train_dataset, val_dataset = fetch_dataloader(args)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(
        nw))  # https://blog.csdn.net/ResumeProject/article/details/125449639
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=nw)

    model = RotationPredictionNet(args, num_classes=args.ori_noise * 2).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4  # 调整正则化强度
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-3,
    #     epochs=args.epochs,
    #     steps_per_epoch=len(train_loader)
    # )

    best_val_mean_err = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, mean_rotation, median_rotation = validate(args, model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step()

        # 模型检查点
        if mean_rotation < best_val_mean_err:
            best_val_mean_err = mean_rotation
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_mean_err
            }, f'saved_model/{args.name}_{start_time}.pth')
            print_colored(f"Saved new best model with mean rotation err: {best_val_mean_err:.4f}", TextColors.BLUE)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"mean rotation: {mean_rotation:.4f}")
        print(f"median rotation: {median_rotation:.4f}")


def test(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = fetch_dataloader(args, split="test")

    model = RotationPredictionNet(args, num_classes=args.ori_noise * 2).to(device)

    model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
    criterion = nn.CrossEntropyLoss()
    val_loss, mean_rotation, median_rotation = validate(args, model, test_loader, criterion, device, vis=False)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"mean rotation: {mean_rotation:.4f}")
    print(f"median rotation: {median_rotation:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=16)


    parser.add_argument('--name', default="14-same-fuse-round-cosine", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=False, action='store_true',
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
    config['restore_ckpt'] = args.restore_ckpt
    config['start_step'] = args.start_step
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    config['train'] = args.train
    config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    print(config)

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if config.dataset == 'vigor':
        print("Dataset is VIGOR!")
    if args.train:
        train(config)
    else:
        test(config)