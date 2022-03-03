import os
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# 用于训练的人脸数据集
class CelebDataset(Dataset):
    def __init__(self):
        self.img_path = './data/celeb/'
        self.imgs = os.listdir(self.img_path)
        # 人脸框顶点坐标（左上右下）
        with open('./data/celeb_rects.json', 'r') as f:
            self.rects = json.load(f)
        # 人脸轮廓点坐标
        with open('./data/celeb_marks.json', 'r') as f:
            self.marks = json.load(f)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = cv2.imread(self.img_path + img_name)
        label = 0   # 标签：0-真人脸、1-假人脸
        if img_name in self.rects:
            l, t, r, b = self.rects[img_name]
            # 一半的概率生成假脸图像
            if random.random() < 0.5:
                label = 1
                mask = np.zeros((218, 178, 3), dtype=np.uint8)
                # ToDo: 更多人脸形状
                # 目前两种mask：矩形、人脸轮廓
                if random.random() < 0.5:
                    l2 = (5 * l + r) // 6
                    t2 = (5 * t + b) // 6
                    r2 = (l + 5 * r) // 6
                    b2 = (t + 5 * b) // 6
                    cv2.rectangle(mask, (l2, t2), (r2, b2), (1, 1, 1), thickness=-1)
                else:
                    cv2.fillConvexPoly(mask, np.array(self.marks[img_name], dtype='int32'), (1, 1, 1))

                scale = random.random() * 1.5 + 0.5
                warped_img = cv2.resize(img, (int(scale * 178), int(scale * 218)))
                warped_img = cv2.GaussianBlur(warped_img, (5, 5), 0)
                warped_img = cv2.resize(warped_img, (178, 218))
                img = mask * warped_img + (1 - mask) * img

            # 切取RoI
            l3 = l - random.random() * (r - l) / 8
            l3 = 0 if l3 < 0 else l3
            t3 = t - random.random() * (b - t) / 5
            t3 = 0 if t3 < 0 else t3
            r3 = r + random.random() * (r - l) / 8
            r3 = 177 if r3 > 177 else r3
            b3 = b + random.random() * (b - t) / 5
            b3 = 217 if b3 > 217 else b3
            img = img[int(t3):int(b3), int(l3):int(r3)]

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        return img.float(), label
