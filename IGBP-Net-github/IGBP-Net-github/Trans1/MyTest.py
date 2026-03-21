# coding=utf-8
import cv2
import numpy as np
import os



cla_path = r'E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB\masks' + '/'  
#E:\yanghe\TransFuse\TransFuse-main\dataset\dataset_offic\dataset\TrainDataset\images
#E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB
masks = os.listdir(cla_path)  
#
num = len(masks)
for index, mask in enumerate(masks):
    img = cv2.imread(cla_path+mask, 0)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 50, 150)  
    kernel = np.ones((4, 4), np.uint8)
    canny = (cv2.dilate(canny, kernel, iterations=1))

    cv2.imwrite(r'E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB\boundary_dilation' + '/' + mask, canny)
    # E:\yanghe\TransFuse\TransFuse-main\dataset\kvasir_clinicDB\Kvasir\boundary_dilation
    # image_path.split('/')[-1]



