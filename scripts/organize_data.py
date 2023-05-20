import os
from shutil import copyfile

root_path = 'I:\Downloads\pathology_data'

destination_folder = '../Data/train_segmentation'
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)

destination_img_folder = os.path.join(destination_folder, 'img')
if not os.path.exists(destination_img_folder):
    os.mkdir(destination_img_folder)

destination_gt_folder = os.path.join(destination_folder, 'gt_raw')
if not os.path.exists(destination_gt_folder):
    os.mkdir(destination_gt_folder)

destination_viz_folder = os.path.join(destination_folder, 'viz')
if not os.path.exists(destination_viz_folder):
    os.mkdir(destination_viz_folder)


for folder in os.listdir(root_path):
    full_folder = os.path.join(root_path, folder)
    if os.path.isdir(full_folder):
        name = folder.split("_")[0] + folder.split("_")[1]
        img_file = os.path.join(full_folder, 'img.png')
        copyfile(img_file, os.path.join(destination_img_folder, name + '.png'))
        gt_file = os.path.join(full_folder, 'label.png')
        copyfile(gt_file, os.path.join(destination_gt_folder, name + '.png'))
        viz_file = os.path.join(full_folder, 'label_viz.png')
        copyfile(viz_file, os.path.join(destination_viz_folder, name + '.png'))
