from PIL import Image
import os, sys
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = 933120000

input_dir = '../Data/train_segmentation/gt_raw'
output_dir = '../Data/train_segmentation/gt'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for filename in os.listdir(input_dir):
    if '.png' not in filename and '.jpg' not in filename:
        continue
    fullname = os.path.join(input_dir, filename)
    gt = Image.open(fullname).convert('L')
    gt_numpy = np.array(gt) * 255
    gt_numpy = np.clip(gt_numpy, 0, 255)
    gt_numpy = gt_numpy.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, filename), gt_numpy)
