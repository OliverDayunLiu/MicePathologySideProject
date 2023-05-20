import cv2
import os, sys
import json
import numpy as np


root_folder = '../data/all_data/2020_12_18'

png_folder = os.path.join(root_folder, 'png')
json_folder = os.path.join(root_folder, 'json')
mask_folder = os.path.join(root_folder, 'mask')

if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)

for filename in os.listdir(png_folder):
    if '.png' not in filename and '.excel' not in filename:
        continue
    print(filename)

    png_file = os.path.join(png_folder, filename)
    png = cv2.imread(png_file)

    mask = np.zeros((png.shape[0], png.shape[1]))

    json_file = os.path.join(json_folder, filename.replace('.png', '.json'))
    if not os.path.exists(json_file):
        print("json file does not exist ", json_file)
        continue
    json_file_handle = open(json_file)
    json_content = json.load(json_file_handle)
    shapes = json_content['shapes'][0]
    points = shapes['points']
    for point in points:
        mask[int(point[1]), int(point[0])] = 255

    mask = cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))

    cv2.imshow("mask", mask)
    cv2.waitKey(0)





