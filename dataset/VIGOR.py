import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.utils.data as data
import cv2
from geometry import get_BEV_tensor, get_BEV_projection
import numpy as np

class VIGOR(Dataset):
    def __init__(self, args, split='train', root='/data/dataset/VIGOR', same_area=True):
        # usr = os.getcwd().split('/')[2]
        # root = os.path.join('/home',usr,root)
        same_area = not args.cross_area

        self.image_size = args.image_size
        # self.start_angle = args.start_angle
        self.fov_size = args.fov_size
        self.bev_size = args.bev_size


        label_root = 'splits'  # 'splits' splits__corrected
        if same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco',
                                    'Chicago']  # ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago'] ['Seattle']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            # original settin is using ['SanFrancisco', 'Chicago'] in test
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        pano_list = []
        pano_label = []
        sat_delta = []

        if split == 'train':
            for city in self.train_city_list:
                label_fname = os.path.join(root, label_root, city, 'same_area_balanced_train.txt'
                if same_area else 'pano_label_balanced.txt')
                with open(label_fname, 'r') as file:
                    for line in file.readlines():
                        data = np.array(line.split(' '))
                        label = []
                        for i in [1, 4, 7, 10]:
                            label.append(os.path.join(root, city, 'satellite', data[i]))
                        delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                        pano_list.append(os.path.join(root, city, 'panorama', data[0]))
                        pano_label.append(label)
                        sat_delta.append(delta)
        else:
            for city in self.test_city_list:
                label_fname = os.path.join(root, label_root, city, 'same_area_balanced_test.txt'
                if same_area else 'pano_label_balanced.txt')
                with open(label_fname, 'r') as file:
                    for line in file.readlines():
                        data = np.array(line.split(' '))
                        label = []
                        for i in [1, 4, 7, 10]:
                            label.append(os.path.join(root, city, 'satellite', data[i]))
                        delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                        pano_list.append(os.path.join(root, city, 'panorama', data[0]))
                        pano_label.append(label)
                        sat_delta.append(delta)

        self.pano_list = pano_list
        self.pano_label = pano_label
        self.sat_delta = sat_delta

        self.split = split
        # self.transform = train_transform(0) if 'augment' in args and args.augment else None
        # self.transform = None
        # self.center = [(self.image_size / 2, self.image_size / 2),
        #                (self.image_size / 2, self.image_size / 2 - self.image_size / 8)] \
        #     if 'orien' in args and args.orien else [(self.image_size // 2.0, self.image_size // 2.0), ]
        pona_path = self.pano_list[0]
        pona = cv2.imread(pona_path, 1)[:, :, ::-1]  # BGR ==> RGB
        self.out = get_BEV_projection(pona, self.image_size, self.image_size, Fov=85 * 2, dty=0, dy=0)
        self.ori_noise = args.ori_noise
        # self.out = None

    def __len__(self):
        return len(self.pano_list)

    def __getitem__(self, idx):
        patch_size = self.image_size
        pona_path = self.pano_list[idx]
        select_ = 0  # random.randint(0,3)
        sat_path = self.pano_label[idx][select_]
        pano_gps = np.array(pona_path[:-5].split(',')[-2:]).astype(float)
        sat_gps = np.array(sat_path[:-4].split('_')[-2:]).astype(float)

        # =================== read satellite map ===================================
        sat = cv2.imread(sat_path, 1)[:, :, ::-1]
        sat = cv2.resize(sat, (patch_size, patch_size))

        # =================== read ground map ===================================
        pano = cv2.imread(pona_path, 1)[:, :, ::-1]

        # =================== get masked pano ===================================
        mask = np.zeros_like(pano)
        h, w, c = pano.shape
        # start_angle = np.random.randint(1, 360-self.fov_size)
        start_angle = 0
        w_start, w_end = w / 360 * start_angle, w / 360 * (start_angle + self.fov_size)
        w_start = int(np.round(w_start))
        w_end = int(np.round(w_end))
        mask[:, w_start:w_end, :] = pano[:, w_start:w_end, :]
        pano = mask

        # rotation_range = self.ori_noise
        # random_ori = np.random.uniform(-1, 1) * rotation_range / 360
        # ori_angle = random_ori * 360
        # pano = np.roll(pano, int(random_ori * pano.shape[1]), axis=1)

        pano_bev = get_BEV_tensor(pano, 500, 500, dty=0, dy=0, out=self.out).numpy().astype(np.uint8)
        # pano_bev = cv2.resize(pano_bev, (patch_size, patch_size))
        pano_bev = cv2.resize(pano_bev, (self.bev_size, self.bev_size))
        bev = torch.from_numpy(pano_bev).float().permute(2, 0, 1)

        sat = torch.from_numpy(sat).float().permute(2, 0, 1)
        pano_gps = torch.from_numpy(pano_gps)  # [batch, 2]
        sat_gps = torch.from_numpy(sat_gps)

        sat_delta_init = torch.from_numpy(self.sat_delta[idx][select_] * patch_size / 640.0).float()
        sat_delta = torch.zeros(2)
        sat_delta[1] = sat_delta_init[0] + patch_size / 2.0
        sat_delta[0] = patch_size / 2.0 - sat_delta_init[1]  # from [y, x] To [x, y], so fit the coord of model out

        # return img1, img2, pano_gps, sat_gps, torch.tensor(ori_angle), sat_delta
        return bev, sat, pano_gps, sat_gps, sat_delta


def fetch_dataloader(args, split='train'):
    train_dataset = VIGOR(args, split)
    print('Training with %d image pairs' % len(train_dataset))

    if split == 'train':
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print("using {} images for training, {} images for validation.".format(train_size, val_size))
        return train_dataset, val_dataset
    else:
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        test_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                      pin_memory=True, shuffle=False, num_workers=nw, drop_last=False)
        return test_loader