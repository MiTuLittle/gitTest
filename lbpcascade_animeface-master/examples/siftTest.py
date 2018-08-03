# -*- coding: utf-8 -*-  
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = cv2.imread('D:\\picSet\\7.jpg')
img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(img_gray,None)
img_sift = np.copy(image)
cv2.drawKeypoints(image,keypoints,img_sift,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Input image',image)
cv2.imshow('SIFT features',img_sift)
cv2.waitKey()
cv2.imread()