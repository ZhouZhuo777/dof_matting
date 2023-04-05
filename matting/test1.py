import numpy
import numpy as np
from PIL import Image
def fun1(png1,png2):
    imgA = Image.open(png1)
    imgB = Image.open(png2)
    width, height = imgA.size
    for x in range(0, width):
        for y in range(0, height):
            color1 = imgA.getpixel((x, y))
            color2 = imgB.getpixel((x, y))
            if color1 == color2:
                imgA.putpixel((x, y), (255, 0, 0,0))
            else:
                imgA.putpixel((x, y), (0,255,0))
    imgA.save('33.png')
    print("1end")


from PIL import Image
from PIL import ImageChops
def fun2(png1,png2):
    imgA = Image.open('png1')
    imgB = Image.open('png2')
    different = ImageChops.difference(imgA, imgB)
    if different.getbbox() is None:
        print('两张图片相同')
    else:
        different.save('44.png')
    print("2end")

from PIL import Image
import math
import operator
from functools import reduce
def fun3(png1,png2):

    image1 = Image.open(png1)
    image2 = Image.open(png2)

    h1 = image1.histogram()
    h2 = image2.histogram()

    result = math.sqrt(reduce(operator.add,  list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1) )
    print("3end")
    return result

png1 = "111.png"
png2 = "222.png"
# fun1(png1,png2)
# fun2(png1,png2)
# result = fun3(png1,png2)
# print(result)


# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity
import imutils
import cv2
import os
import numpy as np

cntsList = []
def JudgmentContains(curCnts):
    for index in range(len(cntsList)):
        (x, y, w, h) = cv2.boundingRect(cntsList[index])
        (x1, y1, w1, h1) = cv2.boundingRect(curCnts)
        if x1 >= x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
            print(11111)
            return True
        if x1 <= x and y1 <= y and x1 + w1 >= x + w and y1+h1 >= y + h:
            cntsList[index] = curCnts
            print(2222)
            return True
    cntsList.append(curCnts)
    return False

# 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
imageA = cv2.imread(png1,cv2.IMREAD_UNCHANGED)
imageB = cv2.imread(png2,cv2.IMREAD_UNCHANGED)
imgB = Image.open(png2)
imgB.save('222_.png')
imageC = cv2.imread("222_.png",cv2.IMREAD_UNCHANGED)
# cv2.imshow("Modified", imageA)
# cv2.waitKey(0)

grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

# 计算两个灰度图像之间的结构相似度指数
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
# print("SSIM:{}".format(score))

# 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite(r"huidu.png", thresh)
cnts,hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv4() else cnts[1]
# print(len(cnts),'\n',hierarchy,type(hierarchy))

if not os.path.exists("./koutu/"):
    os.mkdir("./koutu/")
n=1
curlist = np.array(hierarchy)
# print(curlist.shape)
# mask = np.zeros(imageA.shape, dtype='uint8')

# 找到一系列区域，在区域周围放置矩形
for i in range(0,len(cnts)):
    # if not hierarchy[i][3] == -1:
    #     continue
    area = cv2.contourArea(cnts[i])
    # cv2.arcLength() 轮廓的周长
    if area < 3000:
        continue
    JudgmentContains(cnts[i])

    # (x, y, w, h) = cv2.boundingRect(cnts[i])
    # curImage = cv2.getRectSubPix(imageC,(w,h),(x + w/2,y + h/2))
    # width, height = curImage.size
    # for i in range(0,width):
    #     for j in range(0,height):
    #         # color = curImage.getpixel((i, j))
    #         if curImage.pointPolygonTest(cnts[i],(i,j),True) < 0:
    #             curImage.putpixel((i, j), (0, 0, 0,0))
    # cv2.imwrite("./koutu/"+str(n)+".png", curImage)
    # n+=1
    # cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 10)
    # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 10)
    # cv2.drawContours(imageB, cnts[i], -1, (0, 255, 0), 10) #可以画出轮廓的线
    # center = (int(x),int(y))
    # (x,y),raidus = cv2.minEnclosingCircle(c) #得到外接圆
    # cv2.circle(imageB, center, int(raidus), (0, 255, 0), 10)# 画到外接圆
    # cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
    #可以绘制斜着的框
    # rect = cv2.minAreaRect(cnts[i])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(imageB, [box], 0, (0, 255, 0), 10)

for cnts in cntsList:
    (x, y, w, h) = cv2.boundingRect(cnts)
    curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
    cv2.imwrite("./koutu/" + str(n) + ".png", curImage)
    n += 1
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 10)

# 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
# //cv2.imshow("Modified", imageB)
cv2.imwrite(r"result.png", imageB)
# cv2.imwrite(r"mask.png", mask)
cv2.waitKey(0)

