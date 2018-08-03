#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    global filePath1
    global filePath
    filePath1 = 'D:\\picSet\\cutPic\\result'
    filePath = 'D:\\picSet\\cutPic\\src'

    imgs = os.listdir(filePath)
    imgNum = len(imgs)
    for i_img in range(imgNum):
        picPath = filePath + '\\' + imgs[i_img]
        if os.path.exists(picPath) and len(os.path.basename(picPath).split('.')) > 1:
            findLK_cv2(picPath)
        else:
            print("图片不存在")
            continue


def findLK_plt(picPath):
    flag_1 = 1
    img1 = Image.open(picPath)
    img_gray = np.array(img1.convert('L'))
    cols, rows = img_gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    #开运算
    img_opened = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    img_den = cv2.fastNlMeansDenoising(img_gray, None, 20, 5, 11)
    plt.imshow(img_den)
    plt.show()
    for i in range(1, cols):
        flag_1 = 1
        for j in range(rows):
            if img_den[i, j] > 250:
                continue
            else:
                flag_1 = 2
                break
        if flag_1 == 2:
            break

    plt.imshow(img_gray)
    plt.show()


def findLK_cv2(picPath):
    im = cv2.imread(picPath)
    im2 = im.copy()
    im2 = cv2.GaussianBlur(im2, (3, 3), 0)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #for i_s in range(0,len(cv2.split(im2)) - 1):
    image_s1 = cv2.split(im2)[0]
    image_s2 = cv2.split(im2)[1]
    image_s3 = cv2.split(im2)[2]
    #plt.show()
    #计算单通道图片矩阵的像素平均值
    image_s3_mean = np.mean(image_s3)
    imgray_mean = np.mean(imgray)
    #把像素平均值作为阈值进行处理
    ret, thresh = cv2.threshold(image_s3, image_s3_mean, 255, 0)
    cols, rows = thresh.shape
    #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = thresh[2:cols-200,0:rows]
    image, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im2, contours, -1, (0, 0, 255), 3)
    maxH = 0
    maxW = 0
    x_max = 0
    y_max = 0
    i_max = 0
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(im2, (x, y), (x+w, y+h), (255, 0, 0), 10)
        if x < cols/100 or y < rows/100 or w < cols/100 or h < rows/100:
            continue
        if w > maxW or h > maxH:
            i_max = i
            maxW, maxH, x_max, y_max = w, h, x, y
    if i_max == 0:
        return
    newimage = None
    newimage = im[y_max-2:y_max+maxH, x_max-2:x_max+maxW]
    if newimage is None:
        return
    '''
    #print(contours[i_max])
    plt.imshow(im2)
    plt.show()
    plt.imshow(newimage)
    plt.show()
    '''
    savePath = os.path.basename(picPath).split('.')[0]
    save_filename = '%s-%d.jpg' % (savePath, 1)
    cv2.imwrite(filePath1 + '\\' + save_filename, newimage)

    # 目标图片大小
    dst_w = 700
    dst_h = 800
    # 保存的图片质量
    save_q = 100
    srcPath = filePath1 + '\\' + save_filename
    savePath_1 = filePath1 + '\\' + \
        '%s-%d.jpg' % (os.path.basename(picPath).split('.')[0], 3)
    ImageCompressUtil().resizeImg(
        ori_img=srcPath,
        dst_img=savePath_1,
        dst_w=dst_w,
        dst_h=dst_h,
        save_q=save_q
    )


class ImageCompressUtil(object):
    # 等比例压缩
    def resizeImg(self, **args):
        try:
            args_key = {'ori_img': '', 'dst_img': '',
                        'dst_w': '', 'dst_h': '', 'save_q': 100}
            arg = {}
            for key in args_key:
                if key in args:
                    arg[key] = args[key]
            im = Image.open(arg['ori_img'])
            if im.format in ['gif', 'GIF', 'Gif']:
                return
            ori_w, ori_h = im.size
            widthRatio = heightRatio = None
            ratio = 1
            if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
                if arg['dst_w'] and ori_w > arg['dst_w']:
                    widthRatio = float(arg['dst_w']) / ori_w  # 正确获取小数的方式
                if arg['dst_h'] and ori_h > arg['dst_h']:
                    heightRatio = float(arg['dst_h']) / ori_h
                if widthRatio and heightRatio:
                    if widthRatio < heightRatio:
                        ratio = widthRatio
                    else:
                        ratio = heightRatio
                if widthRatio and not heightRatio:
                    ratio = widthRatio
                if heightRatio and not widthRatio:
                    ratio = heightRatio
                newWidth = int(ori_w * ratio)
                newHeight = int(ori_h * ratio)
            else:
                newWidth = ori_w
                newHeight = ori_h
            if len(im.split()) == 4:
                # prevent IOError: cannot write mode RGBA as BMP
                r, g, b, a = im.split()
                im = Image.merge("RGB", (r, g, b))
            im.resize((newWidth, newHeight), Image.ANTIALIAS).save(
                arg['dst_img'], quality=arg['save_q'])
        except Exception as e:
            return u"压缩失败" + e


if __name__ == '__main__':
    main()
