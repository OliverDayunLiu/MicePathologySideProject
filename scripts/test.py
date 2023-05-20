import cv2
import numpy as np
from PIL import Image


img_path = '../Data/train_segmentation/img/201028162214.png'
img = Image.open(img_path)
#print(img_numpy)
#print(np.max(img_numpy), np.min(img_numpy))
img.show()



img_path = '../Data/train_segmentation/gt/201028162214.png'
img = Image.open(img_path).convert('L')
img_numpy = np.array(img) * 255
img = Image.fromarray(img_numpy)

#print(img_numpy)
#print(np.max(img_numpy), np.min(img_numpy))
img.show()
