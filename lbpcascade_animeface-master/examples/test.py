# -*- coding: utf-8 -*-  
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
filename = 'D:\\picSet\\cutPic\\result\\JT\\1905-1.jpg'
image = cv2.imread(filename)
#边缘填充函数
image1 = cv2.copyMakeBorder(image,200,200,200,200, cv2.BORDER_CONSTANT,value=[255,255,255])

'''
img1 = image
replicate = cv2.copyMakeBorder(img1, 10,100,100,100, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, 100,10,100,100, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, 100,100,10,100, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, 100,100,100,10, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img1, 100,100,100,100, cv2.BORDER_CONSTANT, value=[255,255,255])

plt.subplot(231),plt.imshow(img1),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate),plt.title('REFLECT')
plt.subplot(233),plt.imshow(reflect),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant),plt.title('CONSTANT')
plt.show()
'''

imageBGRA = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
plt.imshow(imageBGRA)
plt.show()
#img = imageBGRA.convert('RGBA')
imageBGRA = Image.fromarray(imageBGRA)
r, g, b, alpha = imageBGRA.split()
alpha = alpha.point(lambda i: i>0 and 178)
imageBGRA.putalpha(alpha)
plt.imshow(imageBGRA)
plt.show()
'''
for i in range(0,799):
    for j in range(0,799):
        if imageGray[i,j] > 200:
            imageGray[i,j] = 0
            image[i,j] = 0
np.savetxt("aa1.txt",imageGray,fmt="%d")
plt.imshow(image)
#a = image.shape[:2]
#p2=cv2.resize(image,(int(a[1]/1.5),int(a[0]/1.5)),interpolation=cv2.INTER_AREA)

cv2.imwrite("faces/" + "7-1.jpg", image)
'''
'''
# RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
B = image[:,:,0]
G = image[:,:,1]
R = image[:,:,2]
# 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。

g = imageGray[:]
p = 0.2989; q = 0.5870; t = 0.1140
B_new = (g-p*R-q*G)/t
B_new = np.uint8(B_new)
src_new = np.zeros((image.shape)).astype("uint8")

src_new[:,:,0] = B_new
src_new[:,:,1] = G
src_new[:,:,2] = R

plt.subplot(133)
plt.imshow(src_new)
'''