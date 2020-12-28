import torch
import os
import numpy as np
import cv2
import json
from PIL import  Image
from tqdm import tqdm

file_path="/data/wurenji/code_new/HRNet-Semantic-Segmentation/data/list/daodixian_seg/test_1209.lst" 
data_path = '/data/wurenji/code_new/HRNet-Semantic-Segmentation/data/daodixian_seg/'
save_dir_now = '/data/wurenji/code_new/HRNet-Semantic-Segmentation/outputs/daodixian_seg_1209/daodixian_seg/seg_hrnet_ocr_w48_epoch48_1209/test_images/'
with open(file_path) as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    image_path = data_path+line
    # print(image_path)
    # break
    img = cv2.imread(image_path)
    # img = Image.open(image_path)
    # print(img)
    img_dst = image_path.split('/')[-1]
    dst_path_image = os.path.join(save_dir_now, img_dst)
    # print('image:',dst_path_image)
    os.makedirs(os.path.dirname(dst_path_image), exist_ok=True)
    cv2.imwrite(dst_path_image, img)

