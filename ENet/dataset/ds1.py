import cv2
import torch

import config as cf
from .augmentation import *
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class dataset_1(Dataset):
    """
    dataset_1 : npy
    """
    def __init__(self,img_npy_path,mask_npy_path,isTrain = True):
        img_npy_path = Path(img_npy_path)
        if img_npy_path.is_file() != True:
            raise ValueError("Path to image npy {} does not exist !".format(img_npy_path))
        mask_npy_path = Path(mask_npy_path)
        if mask_npy_path.is_file() != True:
            raise ValueError("Path to mask npy {} does not exist !".format(mask_npy_path))

        self.imgs = np.load(img_npy_path)
        self.masks = np.load(mask_npy_path)
        # print(self.imgs.shape)
        if isTrain:
            num_train = int(self.imgs.shape[0]*0.8)
            self.imgs = self.imgs[:num_train]
            self.masks = self.masks[:num_train]

        else:
            num_train = int(self.imgs.shape[0]*0.8)
            self.imgs = self.imgs[num_train:]
            self.masks = self.masks[num_train:]

        self.isTrain = isTrain

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img = np.array(img,dtype=np.float32)

        if self.isTrain:
            img,mask = resize(img,mask)
            img = randomHSV(img)
            img,mask = randomHorizontalFlip(img,mask)
            img,mask = randomVerticalFlip(img,mask)
            img,mask = normalize(img,mask)

        else:
            img,mask = resize(img,mask)
            img,mask = normalize(img,mask)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy()).long()
        return img,mask