import argparse
import json
import os

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import time

import torch
import wandb
# import wandb
from easydict import EasyDict
from matplotlib import pyplot as plt, gridspec
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Network import AttentionGridRegistrationNet, GridRegistrationLoss
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
        bev, sat, pano_gps, sat_gps, sat_delta = [x.to(device) for x in data_blob]

        # 可视化输入图像（可选）
        if args.visualize:
            for i in range(8):
                fig = plt.figure(figsize=(12, 5))
                spec = gridspec.GridSpec(1, 2, width_ratios=[300, 512])  # 宽度比例为 300:512

                # BEV 图像
                ax0 = fig.add_subplot(spec[0, 0])
                bev_img = (bev[i] / 255.0).permute(1, 2, 0).cpu().numpy()
                ax0.imshow(bev_img)
                ax0.set_title('BEV Image')

                # Satellite 图像
                ax1 = fig.add_subplot(spec[0, 1])
                sat_img = (sat[i] / 255.0).permute(1, 2, 0).cpu().numpy()
                ax1.imshow(sat_img)

                # 添加散点
                ax1.scatter(
                    sat_delta[i][0].cpu().numpy(),  # sat_delta_x 偏移
                    sat_delta[i][1].cpu().numpy(),  # sat_delta_y 偏移
                    color="red",
                    label="sat_delta"

                )

                # 添加网格线

                grid_size = 4
                step = 512 // grid_size  # 每个网格的间隔 (32 像素)
                for j in range(1, grid_size):
                    # 水平网格线
                    ax1.axhline(y=j * step, color='white', linestyle='--', linewidth=0.5)
                    # 垂直网格线
                    ax1.axvline(x=j * step, color='white', linestyle='--', linewidth=0.5)

                # 设置标题和坐标轴
                ax1.set_title('Satellite Image with Grid')
                ax1.set_xlim(0, 512)  # 保持坐标范围
                ax1.set_ylim(512, 0)  # y 轴翻转，图像原点在左上角
                ax1.axis('on')  # 显示坐标轴

                plt.show()

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        pred_cls, coord_offset = model(sat, bev)

        # 计算损失
        cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
        loss = 1000 * cls_loss + 1 * reg_loss

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
            'cls_loss': f'{cls_loss.item():.4f}',
            'reg_loss': f'{reg_loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

    # 返回平均损失
    return total_loss / len(train_loader)


def validate(args, model, val_loader, criterion, device, vis=False):
    model.eval()
    total_loss = 0
    all_errors = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')

        for i_batch, data_blob in enumerate(pbar):
            # 解包数据
            bev, sat, pano_gps, sat_gps, sat_delta = [x.to(device) for x in data_blob]

            # 前向传播
            pred_cls, coord_offset = model(sat, bev)

            # 计算损失
            cls_loss, reg_loss = criterion(pred_cls, coord_offset, sat_delta)
            loss = 100 * cls_loss + 1 * reg_loss

            # 计算定位误差（像素距离）
            errors = torch.norm(coord_offset.float() - sat_delta.float(), dim=1)
            all_errors.extend(errors.cpu().numpy())

            # 可视化（可选）
            if vis and i_batch % args.vis_freq == 0:
                visualization(bev, sat, coord_offset, sat_delta)

            # 更新进度条
            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'mean_error': f'{errors.mean().item():.2f}px'
            })

    # 计算平均指标
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    overall_mean_error = np.mean(all_errors)
    overall_median_error = np.median(all_errors)

    # 计算不同阈值下的准确率
    thresholds = [5, 10, 25, 50]
    accuracy_at_thresholds = {}
    for threshold in thresholds:
        accuracy = (np.array(all_errors) < threshold).mean()
        accuracy_at_thresholds[f'acc_{threshold}px'] = accuracy * 100

    # 打印验证结果
    print("\nValidation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Mean Error: {overall_mean_error:.2f}px")
    print(f"Median Error: {overall_median_error:.2f}px")
    for threshold, accuracy in accuracy_at_thresholds.items():
        print(f"Accuracy @ {threshold}: {accuracy:.2f}%")

    return avg_loss, overall_mean_error, overall_median_error

def visualization(bev, sat, pred_coords, gt_coords, save_path=None):
    """
    可视化函数，显示预测结果和真实值的对比
    """
    # 将张量转换为numpy数组并反归一化
    bev_img = ((bev[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    sat_img = ((sat[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)

    # 获取预测点和真实点的坐标
    pred_point = pred_coords[0].cpu().numpy()
    gt_point = gt_coords[0].cpu().numpy()

    # 创建图像副本用于绘制
    sat_vis = sat_img.copy()

    # 在卫星图像上绘制预测点（红色）和真实点（绿色）
    cv2.circle(sat_vis, (int(pred_point[0]), int(pred_point[1])), 5, (0, 0, 255), -1)
    cv2.circle(sat_vis, (int(gt_point[0]), int(gt_point[1])), 5, (0, 255, 0), -1)

    # 创建组合图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 显示BEV图像
    ax1.imshow(bev_img)
    ax1.set_title('BEV Image')
    ax1.axis('off')

    # 显示带有标注的卫星图像
    ax2.imshow(sat_vis)
    ax2.set_title('Satellite Image\nRed: Predicted, Green: Ground Truth')
    ax2.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train(args):
    # 设备设置
    device = torch.device("cuda:" + str(args.gpuid[0]) if torch.cuda.is_available() else "cpu")
    start_time = time.strftime('%Y%m%d_%H%M%S')

    # 数据加载
    train_dataset, val_dataset = fetch_dataloader(args)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=nw)

    # 模型初始化
    model = AttentionGridRegistrationNet(args).to(device)

    # 损失函数
    criterion = GridRegistrationLoss(
        grid_size=args.grid_size,
        img_size=args.image_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 学习率
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    # 训练循环
    best_val_mean_err = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # val_loss, mean_error, median_error = validate(args, model, val_loader, criterion, device)

        # 训练
        train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, mean_error, median_error = validate(args, model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step()

        # 模型检查点
        if mean_error < best_val_mean_err:
            best_val_mean_err = mean_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_mean_err
            }, f'location_model/{args.name}_{start_time}.pth')
            print(f"Saved new best model with mean error: {best_val_mean_err:.4f}")

        # 打印训练总结
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean Error: {mean_error:.4f}")
        print(f"Median Error: {median_error:.4f}")

    return model

# def test(args):
#     device = torch.device("cuda:" + str(args.gpuid[0]))
#
#     nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#
#     test_loader = fetch_dataloader(args, split="test")
#
#     model = LocationPredictionNet(args).to(device)
#
#     model, start_epoch, best_val_loss = load_trained_model(model, args.model, device)
#     criterion = nn.CrossEntropyLoss()
#     val_loss, mean_rotation, median_rotation = validate(args, model, test_loader, criterion, device, vis=False)
#
#     print(f"Val Loss: {val_loss:.4f}")
#     print(f"mean rotation: {mean_rotation:.4f}")
#     print(f"median rotation: {median_rotation:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=16)


    parser.add_argument('--name', default="cross-location", help="none")
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
        pass
        # test(config)