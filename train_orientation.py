import argparse
import json
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['WANDB_MODE'] = "offline"
import time

import torch
import wandb
from easydict import EasyDict
from torch import nn, optim
from tqdm import tqdm

from model.orientation_estimator import RotationPredictionNet
from model.loss import  generate_soft_labels, calculate_errors
from utils.util import setup_seed, print_colored, visualization, TextColors
from dataset.VIGOR import fetch_dataloader, VIGOR
import torch.nn.functional as F

def load_trained_model(model, pth_file, device):
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model from {pth_file}, epoch {checkpoint['epoch']}, val_loss {checkpoint['loss']:.4f}")
    return model, checkpoint['epoch'], checkpoint['loss']

def train_epoch(args, model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training')

    for i_batch, data_blob in enumerate(pbar):
        bev, sat, pano_gps, sat_gps, ori_angle, sat_delta, meter_per_pixel, masked_pano, mask, resized_pano, rotated_pano, city, masked_fov = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
        num_class = args.ori_noise * 2

        ori_angle = ori_angle.cpu().numpy()
        rotation_label = np.floor((ori_angle + args.ori_noise) / num_class * (num_class - 1)).astype(
            int)
        rotation_label = torch.tensor(rotation_label, dtype=torch.long, device=bev.device)
        soft_rotation_label = generate_soft_labels(rotation_label, num_class)

        # delta_angle = delta_angle.cpu().numpy()
        # delta_rotation_label = np.floor((delta_angle + args.ori_noise) / num_class * (num_class - 1)).astype(
        #     int)
        # delta_rotation_label = torch.tensor(delta_rotation_label, dtype=torch.long, device=bev.device)
        # print(rotation_label.shape)

        # delta_label = delta_rotation_label - rotation_label

        # soft_delta_label = generate_soft_labels(delta_rotation_label, num_class)

        optimizer.zero_grad()

        pred_label = model(sat, bev)

        log_probs = F.log_softmax(pred_label, dim=1)
        loss = -(soft_rotation_label * log_probs).sum(dim=1).mean()

        # TODO
        # loss = criterion(pred_label, soft_rotation_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            'batch_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (i_batch + 1):.4f}'
        })

        if config['wandb'] and i_batch % 20 == 0:
            wandb.log({'avg_loss': total_loss / (i_batch + 1),
                       })


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

            bev, sat, pano_gps, sat_gps, ori_angle, sat_delta, meter_per_pixel, masked_pano, mask, resized_pano, rotated_pano, city, masked_fov = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in data_blob]
            gt_ori = ori_angle
            ori_angle = ori_angle.cpu().numpy()
            num_class = args.ori_noise * 2

            # rotation_label = np.floor((ori_angle + args.ori_noise) / num_class * (num_class - 1)).astype(int)
            # rotation_label = torch.tensor(rotation_label, dtype=torch.long, device=bev.device)



            pred_label = model(sat, bev)

            err, mean_err, median_err = calculate_errors(gt_ori, pred_label)

            total_mean += mean_err.item()

            all_errors.extend(err.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                # 'val_batch_loss': f'{loss.item():.4f}',
                'mean_rotation_err': f'{mean_err.item():.4f}'
            })

            if vis:
                predicted_labels = torch.argmax(pred_label, dim=1)
                visualization(bev, sat, sat_delta, ori_angle, predicted_labels - 45)

    # 计算平均指标
    avg_loss = total_loss / len(val_loader)
    overall_median = np.median(all_errors)
    overall_mean = np.mean(all_errors)
    if config['wandb'] and i_batch % 20 == 0:
        wandb.log({'overall_median': overall_median,
                   'overall_mean': overall_mean,
                   })

    return avg_loss, overall_mean, overall_median


def train(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))
    start_time = time.strftime('%Y%m%d_%H%M%S')

    # train_dataset, val_dataset = fetch_dataloader(args)
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=nw)
    vigor = VIGOR(args, "train")
    train_loader, val_loader = fetch_dataloader(args, vigor)

    model = RotationPredictionNet(args, num_classes=args.ori_noise * 2).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-5
    )

    best_val_mean_err = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device)

        val_loss, mean_rotation, median_rotation = validate(args, model, val_loader, criterion, device)

        scheduler.step()

        if mean_rotation < best_val_mean_err:
            best_val_mean_err = mean_rotation
            model_path = args.save_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Directory created: {model_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_mean_err
            }, f'{model_path}/{args.name}_{start_time}.pth')
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

    model, start_epoch, best_val_loss = load_trained_model(model, args.orientation_model, device)
    criterion = nn.CrossEntropyLoss()
    val_loss, mean_rotation, median_rotation = validate(args, model, test_loader, criterion, device, vis=False)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"mean rotation: {mean_rotation:.4f}")
    print(f"median rotation: {median_rotation:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config_vigor.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=16)


    parser.add_argument('--name', default="cross-ori45", help="none")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese
    parser.add_argument('--train', default=False)


    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['name'] = args.name
    config['restore_ckpt'] = args.restore_ckpt
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    config['train'] = args.train
    config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if config['wandb']:
        wandb.init(project="g2s-rotation", name=args.name, config=config)

    print(config)

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')


    if args.train:
        train(config)
    else:
        test(config)

    if config['wandb']:
        wandb.finish()
