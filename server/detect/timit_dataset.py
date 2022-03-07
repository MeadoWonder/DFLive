import os
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# 切取RoI
def cut_roi(img, rect):
    height = len(img)
    width = len(img[0])
    l, t, r, b = rect

    l3 = l - random.random() * (r - l) / 8
    l3 = 0 if l3 < 0 else l3
    t3 = t - random.random() * (b - t) / 5
    t3 = 0 if t3 < 0 else t3
    r3 = r + random.random() * (r - l) / 8
    r3 = width - 1 if r3 > width - 1 else r3
    b3 = b + random.random() * (b - t) / 5
    b3 = height - 1 if b3 > height - 1 else b3
    img = img[int(t3):int(b3), int(l3):int(r3)]
    return img


# 用于测试的人脸数据集
class TimitDataset(Dataset):
    def __init__(self):
        self.img_path = './data/timit/'
        self.imgs = os.listdir(self.img_path)
        with open('./data/timit_rects.json', 'r') as f:
            self.rects = json.load(f)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = cv2.imread(self.img_path + img_name)
        label = 0 if 'original' in img_name else 1

        if img_name in self.rects:
            img = cut_roi(img, self.rects[img_name])

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        return img.float(), label
