# -*- coding: utf-8 -*-
import cv2
import os
import matplotlib.pyplot as plt
global img
global point1, point2
global times
times = 1


def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    global times
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        #print(point1)
        cv2.circle(img2, point1, 1, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    # 按住左键拖曳
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        #print(point1)
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        save_filename = '%s-%d.jpg' % (savePath, times)
        cut_img2 = cv2.resize(cut_img, (790, 520),
                              interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(filePath1 + '\\' + save_filename, cut_img2)
        times += 1


def main():
    global img
    global filePath1
    global savePath
    global times
    filePath1 = 'D:\\picSet\\cutPic\\result'
    filePath = 'D:\\picSet\\cutPic\\src'
    imgs = os.listdir(filePath)
    imgNum = len(imgs)
    for j in range(imgNum):
        times = 1
        filename = filePath + '\\' + imgs[j]
        print(filename)
        img = cv2.imread(filename)
        savePath = os.path.basename(filename).split('.')[0]
        cv2.namedWindow('image',0)
        #回调函数
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
