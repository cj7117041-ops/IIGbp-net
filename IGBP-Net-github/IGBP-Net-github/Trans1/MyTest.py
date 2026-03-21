import cv2
import numpy as np
import os

cla_path = r'..root\data\ETIS-LaribPolypDB\masks' + '/'  
masks = os.listdir(cla_path)  # iamges 

num = len(masks)
for index, mask in enumerate(masks):
    img = cv2.imread(cla_path+mask, 0)
    canny = cv2.Canny(img, 50, 150)  
    kernel = np.ones((4, 4), np.uint8)
    canny = (cv2.dilate(canny, kernel, iterations=1))

    cv2.imwrite(r'..root\data\ETIS-LaribPolypDB\boundary_dilation' + '/' + mask, canny)



