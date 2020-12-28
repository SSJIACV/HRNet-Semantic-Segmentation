import os

import cv2
import numpy as np
from PIL import Image

img_mask_dir = '/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/test_results_debug_1217/01a1f5c22461425465d229256c33e01d.png'
img_ori_dir = '/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/test_images_ori/01a1f5c22461425465d229256c33e01d.jpeg'
def draw_contours():
    image_ori = cv2.imread(img_ori_dir)
    image = cv2.imread(img_mask_dir)
    print(image.shape)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
    # cv2.drawContours(image_ori,contours,-1,(0,0,255),3)  
    # cv2.drawContours(image,contours,-1,(0,0,255),3)  
    area = 0
    # contours = np.int0(contours)
    img_result = image_ori.copy()
    for i,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10000:
            print("面积:",area)

            # 最小外接矩形
            # min_rect = cv2.minAreaRect(contour)
            # # print("返回值min_rect:\n", min_rect)
            # print("min_rect shape:",min_rect[0][0])
            # rect_points = cv2.boxPoints(min_rect)
            # # print("返回值rect_points:\n", rect_points)
            # rect_points = np.int0(rect_points)
            # cv2.drawContours(img_result, [rect_points], 0, (0, 0, 255), 2)

            # 直边矩形
            x,y,w,h = cv2.boundingRect(contour)
            # if w*h>200:
            # print('x,y:', 'x', ',', 'y')
            cv2.rectangle(img_result,(x,y),(x+w,y+h),(0,255,0),2)
            
    cv2.imwrite("/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/test_results_debug_1217/res2_boundingbox.png", img_result) 
    # cv2.imwrite("/data/wurenji/code_new/HRNet-Semantic-Segmentation/tools_jss/test_results_debug_1217/res1_1.png", image)   
    cv2.waitKey(0)  

if __name__ == '__main__':
    draw_contours()