#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import cv2


def test_1():
    im = Image.open("D:\\picSet\\cutPic\\src\\3(5).jpg")
    transparent_area = (50, 80, 100, 200)
    transparent = 255  # 用来调透明度，具体可以自己试
    mask = Image.new('L', im.size, color=transparent)
    plt.imshow(mask)
    plt.show()
    draw = ImageDraw.Draw(mask)
    draw.rectangle(transparent_area, fill=0)

    im2 = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2RGBA)
    plt.imshow(im2)
    plt.show()
    im.putalpha(mask)
    plt.imshow(im)
    plt.show()
    im.save('D:\\picSet\\cutPic\\src\\image.png')


def test_2():
    picPath = "D:\\picSet\\cutPic\\src\\038EE1K070-3-2-1.jpg"
    if os.path.exists(picPath):
        im = cv2.imread(picPath)
        im2 = im.copy()
        plt.imshow(im2)
        plt.show()
        image_s1 = cv2.split(im2)[0]
        image_s2 = cv2.split(im2)[1]
        image_s3 = cv2.split(im2)[2]
        pixMean_1 = np.mean(image_s1)
        pixMean_2 = np.mean(image_s2)
        pixMean_3 = np.mean(image_s3)
        plt.subplot(131)
        plt.imshow(image_s1)
        plt.subplot(132)
        plt.imshow(image_s2)
        plt.subplot(133)
        plt.imshow(image_s3)
        plt.show()
        print(pixMean_1,pixMean_2,pixMean_3)
        #canny算子检测边缘（待测试）
        img3 = cv2.GaussianBlur(im2,(5,5),0)
        imgray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(imgray, 50, 100)
        plt.imshow(canny)
        plt.show()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
        thresh_c = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel2)
        thresh_c = cv2.morphologyEx(thresh_c, cv2.MORPH_OPEN, kernel)
        plt.imshow(thresh_c)
        plt.show()

        #获取图片边缘（基本可行）
        imgray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        #plt.imshow(imgray)
        #plt.show()
        im2_PNG = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2RGBA)
        ret, thresh = cv2.threshold(imgray, 235, 255, cv2.THRESH_BINARY)
        #plt.imshow(thresh)
        #plt.show()
        #cols, rows = thresh.shape
        #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
        #thresh = thresh[2:cols-200,0:rows]
        image, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im2, contours, -1, (0, 0, 255), 3)
        plt.imshow(image)
        plt.show()
        mask_inv = cv2.bitwise_not(image)
        #plt.imshow(mask_inv)
        #plt.show()
        img2_bg = cv2.bitwise_and(im2_PNG, im2_PNG, mask=mask_inv)
        plt.imshow(img2_bg)
        plt.show()
        cv2.imwrite("D:\\picSet\\cutPic\\result\\KT\\038EE1K070-3-2-1-3.png", img2_bg)
    else:
        print("图片不存在")
test_2()
