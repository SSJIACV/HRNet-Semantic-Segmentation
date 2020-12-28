import torch
import os
import numpy as np
import cv2
import json
from PIL import  Image
from tqdm import tqdm


# 126fad5c311076ec6a87addc43d909bc.jpeg
image_root = '/data/daodixian_zhixian/images/'
json_path = '/data/daodixian_zhixian/json_segment/job-365971-1.json'
save_dir = '/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/daodixian_seg_vis_no_fill_1208/'
def vis_json_gt(json_path):
    # image_path = os.path.join(image_root,fpath)
    # img = Image.open(image_path)
    # img = np.array(img)
    # h,  w, _  = img.shape
    # print('h,w:',h,w)

    with open(json_path) as f:
        json_gt = json.load(f)

    items = json_gt['items']
    print("num of images", len(items))
    pbar = tqdm(total=len(items))
    count = 0
    for i,item in enumerate(items):
        # pbar.update(1)
        if i%5!=0:
            continue
        if count>20:
            break
        count = count+1
        results = item['results']
        if len(results)==0:
            continue
        pbar.update(1)
        uris = item['uris']
        # if len(uris) != 1:
        #     print(uris)
        uri = uris[0]
        # print(uri)
        fpath = uri.split('/')
        fpath = fpath[1]
        # print(fpath)
        image_path = os.path.join(image_root,fpath)
        img = cv2.imread(image_path)
        # img = Image.open(image_path)
        # img = np.array(img)
        h,  w, _  = img.shape
        print('h,w:',h,w)
        # dst_path = os.path.join(save_dir, fpath)
        # os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # cv2.imwrite(dst_path, image)
        polys = results['polys']
        # points = []
        for poly in polys:
            poly = poly['poly']
            points = []
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
                points.append((point[0],point[1]))
            # print('points:',points)
            points_final = []
            points_final.append(points)
            poly_lines = np.array(points_final, np.int32)
            # print('poly_lines:',poly_lines)
            cv2.polylines(img,poly_lines,1,(255,0,0),4)
            # cv2.fillPoly(img,poly_lines,(255,0,0))
            # break
        dst_path = os.path.join(save_dir, fpath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)


        
if __name__=="__main__":
    vis_json_gt(json_path)