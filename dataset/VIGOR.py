import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split, Subset
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
        self.fov_decay = args.fov_decay
        self.dynamic_fov = args.dynamic_fov
        self.dynamic_low = args.dynamic_low
        self.dynamic_high = args.dynamic_high

        label_root = 'splits__corrected'  # 'splits' splits__corrected
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
                label_fname = os.path.join(root, label_root, city, 'same_area_balanced_train__corrected.txt'
                if same_area else 'pano_label_balanced__corrected.txt')
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
                label_fname = os.path.join(root, label_root, city, 'same_area_balanced_test__corrected.txt'
                if same_area else 'pano_label_balanced__corrected.txt')
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
        from sklearn.utils import shuffle
        for rand_state in range(20):
            self.pano_list, self.pano_label, self.sat_delta = shuffle(self.pano_list, self.pano_label, self.sat_delta,
                                                                      random_state=rand_state)

        self.meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                                     'Seattle': 0.100817 * 640 / 512,
                                     'SanFrancisco': 0.118141 * 640 / 512,
                                     'Chicago': 0.111262 * 640 / 512}

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
        self.masked_fov = 360 - args.fov_size

    def __len__(self):
        return len(self.pano_list)

    def update_fov(self):
        self.fov_size -= self.fov_decay
        print(f"fov update new fov size is: {self.fov_size}")

    def __getitem__(self, idx):
        patch_size = self.image_size
        pano_path = self.pano_list[idx]
        select_ = 0  # random.randint(0,3)
        sat_path = self.pano_label[idx][select_]
        pano_gps = np.array(pano_path[:-5].split(',')[-2:]).astype(float)
        sat_gps = np.array(sat_path[:-4].split('_')[-2:]).astype(float)

        # =================== read satellite map ===================================
        sat = cv2.imread(sat_path, 1)[:, :, ::-1]
        sat = cv2.resize(sat, (patch_size, patch_size))

        # =================== read ground map ===================================
        pano = cv2.imread(pano_path, 1)[:, :, ::-1]

        resized_pano = cv2.resize(pano, (640, 320))
        h, w, c = resized_pano.shape

        # start_angle1 = np.random.randint(0, self.fov_size)
        # w_start1 = int(np.round(w / 360 * start_angle1))
        # w_end1 = int(np.round(w / 360 * (start_angle1 + 360 - self.fov_size)))
        #
        # if start_angle1 > self.masked_fov:
        #     if (start_angle1 + self.masked_fov * 2) < 360:
        #         valid_range = list(range(0, start_angle1 - self.masked_fov)) + list(range(start_angle1 + self.masked_fov, 360 - self.masked_fov))
        #     else:
        #         valid_range = list(range(0, start_angle1 - self.masked_fov))
        # else:
        #     valid_range = list(range(start_angle1 + self.masked_fov, 360 - self.masked_fov))
        #
        # start_angle2 = np.random.choice(valid_range)
        #
        # w_start2 = int(np.floor(w / 360 * start_angle2))
        # w_end2 = int(np.floor(w / 360 * (start_angle2 + 360 - self.fov_size)))

        if self.dynamic_fov:
            masked_fov = 360 - np.random.randint(self.dynamic_low, self.dynamic_high)
        else:
            masked_fov = 360 - self.fov_size

        start_angle1 = np.random.randint(0, 360 - masked_fov)
        w_start1 = int(np.round(w / 360 * start_angle1))
        w_end1 = int(np.round(w / 360 * (start_angle1 + masked_fov)))

        # 创建两个 mask，并应用到原图像上
        mask1 = np.zeros_like(resized_pano)
        mask2 = np.zeros_like(resized_pano)
        pano1 = resized_pano.copy()
        pano2 = resized_pano.copy()
        ones1 = np.ones_like(resized_pano)
        ones2 = np.ones_like(resized_pano)

        pano1[:, w_start1:w_end1, :] = mask1[:, w_start1:w_end1, :]
        # pano2[:, w_start2:w_end2, :] = mask2[:, w_start2:w_end2, :]
        ones1[:, w_start1:w_end1, :] = mask1[:, w_start1:w_end1, :]
        # ones2[:, w_start2:w_end2, :] = mask2[:, w_start2:w_end2, :]
        ones2 = ones2 - ones1
        pano2 = pano2 * ones2

        # =================== get masked pano ===================================
        # mask = np.zeros_like(pano)
        # h, w, c = pano.shape
        # start_angle = np.random.randint(0, 360-self.fov_size)
        # # start_angle = 0
        # w_start, w_end = w / 360 * start_angle, w / 360 * (start_angle + self.fov_size)
        # w_start = int(np.round(w_start))
        # w_end = int(np.round(w_end))
        # mask[:, w_start:w_end, :] = pano[:, w_start:w_end, :]
        # pano = mask

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

        # sat_delta_init = torch.from_numpy(self.sat_delta[idx][select_] * patch_size / 640.0).float()
        # sat_delta = torch.zeros(2)
        # sat_delta[1] = sat_delta_init[0] + patch_size / 2.0
        # sat_delta[0] = patch_size / 2.0 - sat_delta_init[1]  # from [y, x] To [x, y], so fit the coord of model out

        sat_delta_init2 = torch.from_numpy(self.sat_delta[idx][select_] * patch_size / 640.0).float()

        gt_shift_y = sat_delta_init2[0] / 512 * 4  # -L/4 ~ L/4  -1 ~ 1
        gt_shift_x = -sat_delta_init2[1] / 512 * 4  #
        sat_delta = [gt_shift_x, gt_shift_y]

        # plt.figure(figsize=(20, 10))  # 设置图大小
        # plt.imshow(pano1)
        # plt.title(f"pano 1")
        # plt.axis("on")  # 关闭坐标轴
        # plt.show()

        city = ""
        if 'NewYork' in pano_path:
            city = 'NewYork'
        elif 'Seattle' in pano_path:
            city = 'Seattle'
        elif 'SanFrancisco' in pano_path:
            city = 'SanFrancisco'
        elif 'Chicago' in pano_path:
            city = 'Chicago'

        # return img1, img2, pano_gps, sat_gps, torch.tensor(ori_angle), sat_delta
        return bev, sat, pano_gps, sat_gps, torch.tensor(sat_delta, dtype=torch.float32), torch.tensor(
            self.meter_per_pixel_dict[city], dtype=torch.float32), \
            pano1, ones1, pano2, ones2, resized_pano, city, torch.tensor(masked_fov, dtype=torch.float32)


class DistanceBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, train_label):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.backup = []
        self.train_label = train_label

    def check_add(self, id_list, idx):
        '''
        id_list: a list containing grd image indexes we currently have in a batch
        idx: the grd image index to be determined where or not add to the current batch
        '''

        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                if i in sat_idx:
                    return False

        return True

    def __iter__(self):
        batch = []

        for idx in self.sampler:

            if self.check_add(batch, idx):
                # add to batch
                batch.append(idx)

            else:
                # add to back up
                self.backup.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

                remove = []
                for i in range(len(self.backup)):
                    idx = self.backup[i]

                    if self.check_add(batch, idx):
                        batch.append(idx)
                        remove.append(i)

                for i in sorted(remove, reverse=True):
                    self.backup.remove(self.backup[i])

        if len(batch) > 0 and not self.drop_last:
            yield batch
            print('batched all, left in backup:', len(self.backup))

    def __len__(self):

        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def fetch_dataloader(args, vigor=None, split='train'):
    if vigor is None:
        vigor = VIGOR(args, split)

    print('Training with %d image pairs' % len(vigor))

    if split == 'train':

        index_list = np.arange(vigor.__len__())
        train_indices = index_list[0: int(len(index_list) * 0.8)]
        val_indices = index_list[int(len(index_list) * 0.8):]
        training_set = Subset(vigor, train_indices)
        val_set = Subset(vigor, val_indices)

        if not isinstance(vigor.pano_label, np.ndarray):
            vigor.pano_label = np.array(vigor.pano_label)

        train_bs = DistanceBatchSampler(torch.utils.data.RandomSampler(training_set), args.batch_size, True,
                                        vigor.pano_label[np.array(train_indices, dtype=int)])

        train_dataloader = DataLoader(training_set, batch_sampler=train_bs, num_workers=8)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

        print("using {} images for training, {} images for validation.".format(len(training_set), len(val_set)))
        return train_dataloader, val_dataloader
    else:
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        test_loader = data.DataLoader(vigor, batch_size=16,
                                      pin_memory=True, shuffle=False, num_workers=nw, drop_last=False)
        return test_loader
