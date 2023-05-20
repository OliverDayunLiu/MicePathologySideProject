import cv2
import os, sys


tif_folder = '../data/all_data/2020_12_18/tif'
png_folder = '../data/all_data/2020_12_18/png'

if not os.path.exists(png_folder):
    os.mkdir(png_folder)

for filename in os.listdir(tif_folder):
    if '.tif' not in filename and '.excel' not in filename:
        continue
    print(filename)
    tif = cv2.imread(os.path.join(tif_folder, filename))
    cv2.imwrite(os.path.join(png_folder, filename.replace('.tif','.png')), tif)

