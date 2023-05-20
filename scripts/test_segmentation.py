import os, sys
import torchvision.transforms as standard_transforms
import torch
import utils.joint_transforms as joint_transforms
from models.psp_net import PSPNet
import PIL
from PIL import Image
import numpy as np
import cv2

ckpt_path = '../ckpt'
exp_name = 'segmentation'
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def main():

    checkpoint = 'epoch_137_iter_12056_loss_0.03890_lr_0.0035355339.pth'
    folder_to_test = '../data/val_segmentation/img'
    gt_folder = '../data/val_segmentation/gt'
    result_folder = '../results'

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    net = PSPNet(num_classes=2, pretrained=False)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, checkpoint)))
    net.cuda().eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    test_transform = standard_transforms.Compose([
        standard_transforms.Scale(512),
        standard_transforms.ToTensor()
        # standard_transforms.Normalize(*mean_std)
    ])

    for filename in os.listdir(folder_to_test):
        if '.png' not in filename and '.jpg' not in filename:
            continue

        print("Handling file: ", filename)

        fullname = os.path.join(folder_to_test, filename)

        img = Image.open(fullname)
        img = test_transform(img)
        img = img.unsqueeze(0)

        gt = None if gt_folder is None else cv2.imread(os.path.join(gt_folder, filename), 0)
        gt = cv2.resize(gt, (512, 512))
        if gt_folder is not None:
            gt = np.stack([gt, gt, gt], axis=-1)

        input, output = test(img, net)

        input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
        output = np.stack([output, output, output], axis=-1)

        if gt_folder is None:
            combined_img = np.concatenate([input, output], axis=1)
        else:
            combined_img = np.concatenate([input, output, gt], axis=1)

        cv2.imwrite(os.path.join(result_folder, filename), combined_img)

def test(img, net):

    input = img.cuda()
    output = net(input)

    input = input[0].cpu().detach() # between 0 and 1
    input = input.permute((1,2,0))
    input = input * 255
    input = input.numpy()
    input = np.clip(input, 0, 255)
    input = input.astype(np.uint8)

    output_reformat = output[0]  # BS is 1, so get the element directly
    class_predictions = torch.argmax(output_reformat, dim=0).float().cpu().detach() # 512, 512

    output = class_predictions * 255
    output = output.numpy()
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return input, output


if __name__ == '__main__':
    main()
