import argparse
import json
import os
import time

import torch
# import wandb
from easydict import EasyDict
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from model.Network import RotationPredictionNet
from model.loss import RotationLoss
from utils.util import setup_seed, print_colored, count_parameters
from dataset.VIGOR import fetch_dataloader


def train(args):
    device = torch.device("cuda:" + str(args.gpuid[0]))
    model = None

    # model = nn.DataParallel(model, device_ids=args.gpuid).to(device)
    # print("Parameter Count: %d" % count_parameters(model))
    model = RotationPredictionNet(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = RotationLoss()

    train_dataset, val_dataset = fetch_dataloader(args)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(
        nw))  # https://blog.csdn.net/ResumeProject/article/details/125449639
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=nw)
    for i_batch, data_blob in enumerate(train_loader):
        optimizer.zero_grad()
        bev, sat, grd_gps, sat_gps, ori_angle, sat_delta = [x.to(device) for x in data_blob]  # img1, img2, pona_gps, sat_gps
        pred_angle = model(sat, bev)

        # 计算损失
        loss = criterion(pred_angle, ori_angle)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dataset/config.json", type=str, help="path of config file")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])

    parser.add_argument('--name', default="bais's affection", help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--cross_area', default=True, action='store_true',
                        help='Cross_area or same_area')  # Siamese

    parser.add_argument('--best_dis', type=float, default=1e8)

    args = parser.parse_args()

    config = json.load(open(args.config ,'r'))
    config = EasyDict(config)
    config['config'] = args.config
    config['best_dis'] = args.best_dis
    config['validation'] = args.validation
    config['name'] = args.name
    config['restore_ckpt'] = args.restore_ckpt
    config['start_step'] = args.start_step
    config['gpuid'] = args.gpuid
    config['cross_area'] = args.cross_area
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # wandb.init(project="multi-local-rotation", name=args.name, config=config)
    print(config)

    setup_seed(2023)
    print_colored(time.strftime('%Y-%m-%d %H:%M:%S' ,time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if config.dataset == 'vigor':
        print("Dataset is VIGOR!")

    train(config)