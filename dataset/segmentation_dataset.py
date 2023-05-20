from torch.utils.data import Dataset
import os, sys
from PIL import Image
import numpy as np
import torch

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, shared_transform, img_transform, gt_transform):
        self.root_dir = root_dir
        self.shared_transform = shared_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.img_paths = []
        self.gt_paths = []

        img_folder = os.path.join(self.root_dir, 'img')
        for file in os.listdir(img_folder):
            if '.png' not in file and '.jpg' not in file:
                continue
            filename = os.path.join(img_folder, file)
            self.img_paths.append(filename)

        gt_folder = os.path.join(self.root_dir, 'gt')
        for file in os.listdir(gt_folder):
            if '.png' not in file and '.jpg' not in file:
                continue
            filename = os.path.join(gt_folder, file)
            self.gt_paths.append(filename)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        gt = Image.open(self.gt_paths[idx]).convert('L')
        img, gt = self.shared_transform(img, gt)
        img = self.img_transform(img)
        gt = self.gt_transform(gt) # a 2d image. Will reformat a new depth dimension. Each d in D means a class. Range: 0 - 1  size: 1 x 512 x 512
        gt_reformatted = torch.zeros((2, gt.size(1), gt.size(2)))
        gt_reformatted[0, :, :][gt[0] < 0.5] = 1
        gt_reformatted[1, :, :][gt[0] >= 0.5] = 1
        return img, gt_reformatted
