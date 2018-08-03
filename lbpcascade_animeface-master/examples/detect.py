# -*- coding: utf-8 -*-
import cv2
import random
from PIL import Image
from PIL import ImageFile
import imghdr
import sys
import os.path
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def detect(filename, cascade_file="d:/VSWorkSpace/python/lbpcascade_animeface-master/lbpcascade_animeface-master/haarcascade_frontalface_alt2.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    #haarcascade_frontalface_alt2.xml
    cascade = cv2.CascadeClassifier(cascade_file)
    # 这一段是为了防止数据集中有非jpg的图像，如果有，即转换一下格式
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if imghdr.what(filename) == "png":
        Image.open(filename).convert("RGB").save(filename)
    ######
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('shape of image is {}'.format(gray.shape))
    print('Dimension of image is {}'.format(gray.ndim))
    print('Max of image is {}'.format(gray.max()))
    print('Min of image is {}'.format(gray.min()))
    plt.imshow(gray)
    plt.show()

    def gray_2(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_3 = gray_2(image)
    #gray = cv2.equalizeHist(gray)
    print('shape of image is {}'.format(gray_3.shape))
    print('Dimension of image is {}'.format(gray_3.ndim))
    print('Max of image is {}'.format(gray_3.max()))
    print('Min of image is {}'.format(gray_3.min()))
    plt.imshow(gray_3)
    plt.show()
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.08,
                                     minNeighbors=4,
                                     minSize=(24, 24))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w, :]
        #face = cv2.resize(face, (96, 96))
        plt.imshow(face)
        plt.show()
        save_filename = '%s-%d.jpg' % (
            os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("faces/" + save_filename, face)

#if len(sys.argv) != 2:
#    sys.stderr.write("usage: detect.py <filename>\n")
#    sys.exit(-1)


if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('D:\\picSet\\*.jpg')
    for filename in file_list:
        detect(filename)
