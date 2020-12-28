"""
This python script is used for 将绝缘子分割的json标注转换为png标注，同时进行训练集与验证集的划分
"""
# @Time    : 2020/12/09 18:02
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : json2lableImg_jss.py


import torch
import os
import numpy as np
import cv2
import json
from PIL import  Image
from tqdm import tqdm
import PIL.ImageDraw as ImageDraw
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])


image_root = '/data/daodixian_seg/images/'
json_path = '/data/daodixian_seg/json_segment/job-365971-1.json'
save_dir = '/data/daodixian_seg/'   # /train_1209/annotations  /val_1209/annotations

def json2lableImg(json_path,encoding="color"):
# def json2lableImg(json_path,encoding='ids'):
    with open(json_path) as f:
        json_gt = json.load(f)
    # the background
    if encoding == "ids":
        background = 0
    elif encoding == "color":
        background = (  0,  0,  0)
    
    items = json_gt['items']
    print("num of images", len(items))
    pbar = tqdm(total=len(items))
    count = 0
    lst_train = []
    lst_val = []
    for i,item in enumerate(items):
        results = item['results']
        if len(results)==0:
            continue
        count = count+1
        pbar.update(1)
        uris = item['uris']
        uri = uris[0]
        fpath = uri.split('/')
        fpath = fpath[1]
        fpath_labelImg = fpath.replace( ".jpeg" , ".png" )
        # print(fpath)
        if count%8==0:
            save_dir_now = os.path.join(save_dir, 'val_1209')
            lst_save1 = os.path.join('val_1209','images',fpath)
            lst_save2 = os.path.join('val_1209','annotations',fpath_labelImg)
            lst_val.append(lst_save1 + '  ' + lst_save2)
        else:
            save_dir_now = os.path.join(save_dir, 'train_1209')
            lst_save1 = os.path.join('train_1209','images',fpath)
            lst_save2 = os.path.join('train_1209','annotations',fpath_labelImg)
            lst_train.append(lst_save1 + '  ' + lst_save2)
        image_path = os.path.join(image_root,fpath)
        img = cv2.imread(image_path)
        h,  w, _  = img.shape
        size = (w, h)
        # # print('h,w:',h,w)
        # dst_path_image = os.path.join(save_dir_now, 'images', fpath)
        # # print('image:',dst_path_image)
        # os.makedirs(os.path.dirname(dst_path_image), exist_ok=True)
        # cv2.imwrite(dst_path_image, img)
        # cv2.imwrite('/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/1.jpeg',img)
        # dst_path_labelImg = os.path.join(save_dir_now, 'annotations', fpath_labelImg)
        dst_path_labelImg = os.path.join(save_dir_now, 'annotations','annotations_color', fpath_labelImg)  # 画color图时
        # print('label:',dst_path_labelImg)

        # if count>10:
        #     break

        # this is the image that we want to create
        if encoding == "color":
            labelImg = Image.new("RGBA", size, background)
        else:
            labelImg = Image.new("L", size, background)
        drawer = ImageDraw.Draw( labelImg )
        polys = results['polys']
        for poly in polys:
            poly = poly['poly']
            lable = 'jueyuanzi'
            if encoding == "ids":
                val = 1   # lable id 
            elif encoding == "color":
                val = (220, 20, 60)
            polygon = []
            for point in poly:
                if point[0]>1 :
                    point[0] = 1
                if point[0]<0:
                    point[0] = 0
                if point[1]>1 :
                    point[1] = 1
                if point[1]<0:
                    point[1] = 0
                point[0] = point[0]*w
                point[1] = point[1]*h
                point_new = Point(point[0],point[1])
                polygon.append(point_new)
            drawer.polygon( polygon, fill=val )
        # labelImg.save('/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/1.png')
        labelImg.save(dst_path_labelImg)

    # f=open("/data/daodixian_zhixian/train_1209.lst","w")  # /data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss
    # # f=open("/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/train_1209.lst","w")
    # for item in lst_train:
    #     f.write(item+'\n')
    # f.close()
    # f1=open("/data/daodixian_zhixian/val_1209.lst","w")
    # # f1=open("/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/val_1209.lst","w")
    # for item in lst_val:
    #     f1.write(item+'\n')
    # f1.close()
if __name__=="__main__":
    json2lableImg(json_path)