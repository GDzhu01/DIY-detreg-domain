import cv2, math
import numpy as np
import os
list = ['zhuanyitest']
myPath = 'zhuanyitest'
outPath = 'finet/train/image'

def demo(P, outP, name,pro):
    img = cv2.imread(os.path.join(P,name))
    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.6  # 亮度
    beta = 0.07  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    cv2.imwrite(outP +'/w'+ name, img_f*255)
    #cv2.imshow("src", img)
    #cv2.imshow("dst", img_f)
    #cv2.waitKey()

def run():
    #切换到源目录，遍历目录下所有图片
    for j in list:
        for i in os.listdir(j):
            #检查后缀
            postfix = os.path.splitext(i)[1]
            #print(postfix,i)
            if postfix == '.jpg' or postfix == '.png':
                demo(j, outPath, i, postfix)
if __name__ == '__main__':
    run()