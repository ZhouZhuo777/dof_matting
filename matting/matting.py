import base64

from psd_tools import PSDImage


# def draw_rect(size, save_path):
#     imgbase = Image.open("test.png")
#     base_width, base_height = imgbase.size
#     img_new = Image.new('RGBA', size)
#     new_width, new_height1 = size
#
#     base_halfW = int(base_width / 2)
#     base_halfH = int(base_height / 2)
#
#     redW = new_width - base_width
#     redH = new_height1 - base_height
#
#     DW = new_width - base_halfW
#     DH = new_height1 - base_halfH
#
#     for x in range(0, base_halfW):
#         for y in range(0, base_halfH):
#             img_new.putpixel((x, y), imgbase.getpixel((x, y)))
#     for x in range(base_halfW, base_width):
#         for y in range(0, base_halfH):
#             img_new.putpixel((x + redW, y), imgbase.getpixel((x, y)))
#     for x in range(0, base_halfW):
#         for y in range(base_halfH, base_height):
#             img_new.putpixel((x, y + redH), imgbase.getpixel((x, y)))
#     for x in range(base_halfW, base_width):
#         for y in range(base_halfH, base_height):
#             img_new.putpixel((x + redW, y + redH), imgbase.getpixel((x, y)))
#     for x in range(base_halfW, DW):
#         for y in range(0, 42):
#             img_new.putpixel((x, y), (95, 235, 95, 255))
#     for x in range(base_halfW, DW):
#         for y in range(new_height1 - 42, new_height1):
#             img_new.putpixel((x, y), (95, 235, 95, 255))
#     for x in range(0, 42):
#         for y in range(base_halfH, DH):
#             img_new.putpixel((x, y), (95, 235, 95, 255))
#     for x in range(new_width - 42, new_width):
#         for y in range(base_halfH, DH):
#             img_new.putpixel((x, y), (95, 235, 95, 255))
#
#     img_new.save(save_path)


# draw_rect((3000,2000),"999999.png")

def fun2(png1, png2):
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


def fun3(png1, png2):
    image1 = Image.open(png1)
    image2 = Image.open(png2)

    h1 = image1.histogram()
    h2 = image2.histogram()

    result = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    print("3end")
    return result


# fun1(png1,png2)
# fun2(png1,png2)
# result = fun3(png1,png2)
# print(result)

# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity
from PIL import ImageChops
from PIL import Image
import math
import operator
from functools import reduce
# -*- coding: utf-8 -*-
import skimage
# from skimage.metrics import structural_similarity
import imutils
import cv2
import os
from pathlib import Path
import numpy as np


class AutoMatting():
    def __init__(self, png1, png2, outpath):
        self.png1 = png1
        self.png2 = png2
        # self.baseframepng = "G:\\img_lib/frame_base.png"
        self.cntsList = []
        self.minArea = 1000
        self.minW = 300
        self.minH = 380
        self.minR = 190
        self.outpath = outpath
        self.px2cm = 1 / 28.35

        from frame_base_png import img as frame_base
        tmp = open('frame_base.png', 'wb')
        tmp.write(base64.b64decode(frame_base))
        tmp.close()
        self.baseframepng = "frame_base.png"

    def JudgmentContains(self, curCnts):
        for index in range(len(self.cntsList)):
            (x, y, w, h) = cv2.boundingRect(self.cntsList[index])
            (x1, y1, w1, h1) = cv2.boundingRect(curCnts)
            if x1 >= x and y1 >= y and x1 + w1 <= x + w and y1 + h1 <= y + h:  # 传入的框被之前的框包含
                return
            elif x1 <= x and y1 <= y and x1 + w1 >= x + w and y1 + h1 >= y + h:  # 传入的框包含之前的框
                self.cntsList[index] = None
                if not self.JudgmentList(curCnts):
                    self.cntsList.append(curCnts)
            elif (x1 + w1 >= x and x + w >= x1 and y1 + h1 >= y and y + h >= y1):  # 传入的框相交之前的框
                s = w * h
                s1 = w1 * h1
                if s1 > s:
                    self.cntsList[index] = None
                    if not self.JudgmentList(curCnts):
                        self.cntsList.append(curCnts)
                else:
                    return
        if not self.JudgmentList(curCnts):
            self.cntsList.append(curCnts)

    def JudgmentList(self, curCnts):
        (x, y, w, h) = cv2.boundingRect(curCnts)
        for cnt in self.cntsList:
            (x1, y1, w1, h1) = cv2.boundingRect(cnt)
            if (x, y, w, h) == (x1, y1, w1, h1):
                return True
        return False

    def play(self):
        print("开始抠图" + self.outpath)
        splitPath = self.png1.split("\\")
        # outPutPath = self.png1.replace(splitPath[len(splitPath)-1],"koutu/")
        outPutPath = self.outpath
        a = Path(outPutPath)
        # a = Path("./koutu/")
        a.mkdir(exist_ok=True)

        # 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
        imageA = cv2.imread(self.png1, cv2.IMREAD_UNCHANGED)
        imageB = cv2.imread(self.png2, cv2.IMREAD_UNCHANGED)
        imgB = Image.open(self.png2)
        imgB.save(outPutPath + 'back_up.png')
        imageC = cv2.imread(outPutPath + "back_up.png", cv2.IMREAD_UNCHANGED)
        grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        self.wholeH, self.wholeW, self.channels = imageA.shape

        # 计算两个灰度图像之间的结构相似度指数
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM:{}".format(score))

        # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imwrite(outPutPath + "huidu.png", thresh) #保存灰度图
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv4() else cnts[1]
        # print(len(cnts),'\n',hierarchy,type(hierarchy))

        # mask = np.zeros(imageA.shape, dtype='uint8')

        # 找到一系列区域，在区域周围放置矩形
        for i in range(0, len(cnts)):
            # if not hierarchy[i][3] == -1:
            #     continue
            area = cv2.contourArea(cnts[i])
            # cv2.arcLength() 轮廓的周长
            if area < self.minArea:
                continue
            self.JudgmentContains(cnts[i])

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
            # 可以绘制斜着的框
            # rect = cv2.minAreaRect(cnts[i])
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(imageB, [box], 0, (0, 255, 0), 10)
        n = 1
        # if not os.path.exists("./koutu/"):
        #     os.mkdir("./koutu/")

        filename = outPutPath + "coordinate"
        f = open(filename, "w")
        f.write(f"{round(self.wholeW * self.px2cm, 3)},{round(self.wholeH * self.px2cm, 3)}" + '\n')
        self.index = 1
        for cnts in self.cntsList:
            if cnts is None:
                continue
            (x, y, w, h) = cv2.boundingRect(cnts)  # 得到外接矩形
            (c_x, c_y), raidus = cv2.minEnclosingCircle(cnts)  # 得到外接圆
            curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
            cv2.imwrite(outPutPath + "mix_" + str(n) + ".png", curImage)
            self.get_location_file(x + w / 2, y + h / 2, w, h, f)

            center = (int(c_x), int(c_y))
            r = int(raidus)
            rectS = w * h
            rectC = raidus * raidus * math.pi

            oldw = w
            oldh = h
            oldr = r

            if r < 190: r = 190
            centerCut = (r, r)
            if w < h:
                if w < self.minW: w = self.minW
                if h < self.minH: h = self.minH
            else:
                if h < self.minW: h = self.minW
                if w < self.minH: w = self.minH
            if rectS > rectC:
                # cv2.circle(imageB, center, r, (95,235,95), 10)  # 画外接圆
                cv2.circle(imageB, center, oldr, (95, 235, 95), 10)  # 画老的，小的外接圆
                imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                imageCircle[:] = (0, 0, 0, 0)
                cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 42)  # 画每个抠图的圆边框
                cv2.imwrite(outPutPath + "mix_" + str(n) + "_frame.png", imageCircle)

            else:
                # cv2.rectangle(imageB, (x, y), (x + w, y + h), (95,235,95), 10)  # 画外接矩形
                cv2.rectangle(imageB, (x, y), (x + oldw, y + oldh), (95, 235, 95), 10)  # 画老的小的外接矩形
                # imageRect = np.zeros((curImage.shape[0], curImage.shape[1], 4))  # 创建opencv图像
                # imageRect[:] = (0, 0, 0, 0)
                self.draw_rect((w, h), outPutPath + "mix_" + str(n) + "_frame.png")
                # cv2.rectangle(imageRect, (21, 21), (w - 21, h - 21), (95,235,95, 255), 42)  # 画每个抠图的边框
                # cv2.imwrite(outPutPath + str(n) + "_rect.png", imageRect)
            # print(curImage.size)
            n += 1
        f.close()
        cv2.imwrite(outPutPath + "huidu.png", thresh)  # 保存灰度图
        cv2.imwrite(outPutPath + "result.png", imageB)  # 得到原图的画框图

        # imageRect = np.zeros((500,500,4))
        # imageRect[:] = (0, 0, 0, 0)
        # cv2.imshow("Modified", imageRect)
        # 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
        # //cv2.imshow("Modified", imageB)
        # cv2.imwrite(r"mask.png", mask)
        import os
        # if a.is_dir():
        #     os.startfile(outPutPath)
        os.remove(outPutPath + 'back_up.png')
        print(outPutPath + "抠图结束")
        os.remove('frame_base.png')
        # cv2.waitKey(0)

    def play1(self):
        self.minArea = 500
        print("开始抠图" + self.outpath)
        splitPath = self.png1.split("\\")
        # outPutPath = self.png1.replace(splitPath[len(splitPath)-1],"koutu/")
        outPutPath = self.outpath
        a = Path(outPutPath)
        # a = Path("./koutu/")
        a.mkdir(exist_ok=True)

        # 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
        imageA = cv2.imread(self.png1, cv2.IMREAD_UNCHANGED)
        imageB = cv2.imread(self.png2, cv2.IMREAD_UNCHANGED)
        imgB = Image.open(self.png2)
        imgB.save(outPutPath + 'back_up.png')
        imageC = cv2.imread(outPutPath + "back_up.png", cv2.IMREAD_UNCHANGED)
        grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        self.wholeH, self.wholeW, self.channels = imageA.shape

        # 计算两个灰度图像之间的结构相似度指数
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM:{}".format(score))

        # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imwrite(outPutPath + "huidu.png", thresh) #保存灰度图
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv4() else cnts[1]
        # print(len(cnts),'\n',hierarchy,type(hierarchy))

        # mask = np.zeros(imageA.shape, dtype='uint8')

        # 找到一系列区域，在区域周围放置矩形
        for i in range(0, len(cnts)):
            # if not hierarchy[i][3] == -1:
            #     continue
            area = cv2.contourArea(cnts[i])
            # cv2.arcLength() 轮廓的周长
            if area < self.minArea:
                continue
            self.JudgmentContains(cnts[i])

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
            # 可以绘制斜着的框
            # rect = cv2.minAreaRect(cnts[i])
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(imageB, [box], 0, (0, 255, 0), 10)
        n = 1
        # if not os.path.exists("./koutu/"):
        #     os.mkdir("./koutu/")

        filename = outPutPath + "coordinate"
        f = open(filename, "w")
        f.write(f"{round(self.wholeW * self.px2cm, 3)},{round(self.wholeH * self.px2cm, 3)}" + '\n')
        self.index = 1
        for cnts in self.cntsList:
            if cnts is None:
                continue
            (x, y, w, h) = cv2.boundingRect(cnts)  # 得到外接矩形
            (c_x, c_y), raidus = cv2.minEnclosingCircle(cnts)  # 得到外接圆
            curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
            cv2.imwrite(outPutPath + "mix_" + str(n) + ".png", curImage)
            self.get_location_file(x + w / 2, y + h / 2, w, h, f)

            center = (int(c_x), int(c_y))
            r = int(raidus)
            rectS = w * h
            rectC = raidus * raidus * math.pi

            oldw = w
            oldh = h
            oldr = r

            if r < 190: r = 190
            centerCut = (r, r)
            if w < h:
                if w < self.minW: w = self.minW
                if h < self.minH: h = self.minH
            else:
                if h < self.minW: h = self.minW
                if w < self.minH: w = self.minH
            if rectS > rectC:
                # cv2.circle(imageB, center, r, (95,235,95), 10)  # 画外接圆
                cv2.circle(imageB, center, oldr, (95, 235, 95), 10)  # 画老的，小的外接圆
                imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                imageCircle[:] = (0, 0, 0, 0)
                cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 42)  # 画每个抠图的圆边框
                cv2.imwrite(outPutPath + "mix_" + str(n) + "_frame.png", imageCircle)

            else:
                # cv2.rectangle(imageB, (x, y), (x + w, y + h), (95,235,95), 10)  # 画外接矩形
                cv2.rectangle(imageB, (x, y), (x + oldw, y + oldh), (95, 235, 95), 10)  # 画老的小的外接矩形
                # imageRect = np.zeros((curImage.shape[0], curImage.shape[1], 4))  # 创建opencv图像
                # imageRect[:] = (0, 0, 0, 0)
                self.draw_rect((w, h), outPutPath + "mix_" + str(n) + "_frame.png")
                # cv2.rectangle(imageRect, (21, 21), (w - 21, h - 21), (95,235,95, 255), 42)  # 画每个抠图的边框
                # cv2.imwrite(outPutPath + str(n) + "_rect.png", imageRect)
            # print(curImage.size)
            n += 1
        f.close()
        cv2.imwrite(outPutPath + "huidu.png", thresh)  # 保存灰度图
        cv2.imwrite(outPutPath + "result.png", imageB)  # 得到原图的画框图

        # imageRect = np.zeros((500,500,4))
        # imageRect[:] = (0, 0, 0, 0)
        # cv2.imshow("Modified", imageRect)
        # 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
        # //cv2.imshow("Modified", imageB)
        # cv2.imwrite(r"mask.png", mask)
        import os
        # if a.is_dir():
        #     os.startfile(outPutPath)
        os.remove(outPutPath + 'back_up.png')
        print(outPutPath + "抠图结束")
        os.remove('frame_base.png')
        # cv2.waitKey(0)

    def play2(self):
        print("开始抠图" + self.outpath)
        splitPath = self.png1.split("\\")
        # outPutPath = self.png1.replace(splitPath[len(splitPath)-1],"koutu/")
        outPutPath = self.outpath
        a = Path(outPutPath)
        # a = Path("./koutu/")
        a.mkdir(exist_ok=True)

        # 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
        imageA = cv2.imread(self.png1, cv2.IMREAD_UNCHANGED)
        imageB = cv2.imread(self.png2, cv2.IMREAD_UNCHANGED)
        imgB = Image.open(self.png2)
        imgB.save(outPutPath + 'back_up.png')
        imageC = cv2.imread(outPutPath + "back_up.png", cv2.IMREAD_UNCHANGED)
        # grayA = cv2.cvtColor(imageA, cv2.COLOR_RGBA2GRAY)
        # grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        self.wholeH, self.wholeW, self.channels = imageB.shape

        if self.wholeW<1600:
            self.minW = self.minW/2
            self.minH = self.minH/2
            self.minR = self.minR/2

        # 计算两个灰度图像之间的结构相似度指数
        # (score, diff) = structural_similarity(grayA, grayB, full=True)
        # diff = (diff * 255).astype("uint8")
        # print("SSIM:{}".format(score))

        # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
        # thresh = cv2.threshold(imageA, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imwrite(outPutPath + "huidu.png", thresh) #保存灰度图
        cnts, hierarchy = cv2.findContours(imageA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv4() else cnts[1]
        # print(len(cnts),'\n',hierarchy,type(hierarchy))

        # mask = np.zeros(imageA.shape, dtype='uint8')

        # 找到一系列区域，在区域周围放置矩形
        for i in range(0, len(cnts)):
            # if not hierarchy[i][3] == -1:
            #     continue
            area = cv2.contourArea(cnts[i])
            # cv2.arcLength() 轮廓的周长
            # if area < self.minArea:
            #     continue
            self.JudgmentContains(cnts[i])

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
            # 可以绘制斜着的框
            # rect = cv2.minAreaRect(cnts[i])
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(imageB, [box], 0, (0, 255, 0), 10)
        n = 1
        # if not os.path.exists("./koutu/"):
        #     os.mkdir("./koutu/")

        filename = outPutPath + "coordinate"
        f = open(filename, "w")
        f.write(f"{round(self.wholeW * self.px2cm, 3)},{round(self.wholeH * self.px2cm, 3)}" + '\n')
        self.index = 1
        for cnts in self.cntsList:
            if cnts is None:
                continue
            (x, y, w, h) = cv2.boundingRect(cnts)  # 得到外接矩形
            (c_x, c_y), raidus = cv2.minEnclosingCircle(cnts)  # 得到外接圆
            curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
            cv2.imwrite(outPutPath + "mix_" + str(n) + ".png", curImage)
            self.get_location_file(x + w / 2, y + h / 2, w, h, f)

            center = (int(c_x), int(c_y))
            r = int(raidus)
            rectS = w * h
            rectC = raidus * raidus * math.pi

            oldw = w
            oldh = h
            oldr = r

            if r < self.minR:
                # print(r)
                r = self.minR
            centerCut = (r, r)
            if w < h:
                if w < self.minW: w = self.minW
                if h < self.minH: h = self.minH
            else:
                if h < self.minW: h = self.minW
                if w < self.minH: w = self.minH
            if rectS > rectC:
                cv2.circle(imageB, center, r, (95,235,95), 10)  # 画外接圆
                # cv2.circle(imageB, center, oldr, (95, 235, 95), 10)  # 画老的，小的外接圆
                imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                imageCircle[:] = (0, 0, 0, 0)
                cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 42)  # 画每个抠图的圆边框
                cv2.imwrite(outPutPath + "mix_" + str(n) + "_frame.png", imageCircle)

            else:
                cv2.rectangle(imageB, (x, y), (x + w, y + h), (95,235,95), 10)  # 画外接矩形
                # cv2.rectangle(imageB, (x, y), (x + oldw, y + oldh), (95, 235, 95), 10)  # 画老的小的外接矩形
                # imageRect = np.zeros((curImage.shape[0], curImage.shape[1], 4))  # 创建opencv图像
                # imageRect[:] = (0, 0, 0, 0)
                self.draw_rect((w, h), outPutPath + "mix_" + str(n) + "_frame.png")
                # cv2.rectangle(imageRect, (21, 21), (w - 21, h - 21), (95,235,95, 255), 42)  # 画每个抠图的边框
                # cv2.imwrite(outPutPath + str(n) + "_rect.png", imageRect)
            # print(curImage.size)
            n += 1
        f.close()
        # cv2.imwrite(outPutPath + "huidu.png", thresh)  # 保存灰度图
        cv2.imwrite(outPutPath + "result.png", imageB)  # 得到原图的画框图

        # imageRect = np.zeros((500,500,4))
        # imageRect[:] = (0, 0, 0, 0)
        # cv2.imshow("Modified", imageRect)
        # 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
        # //cv2.imshow("Modified", imageB)
        # cv2.imwrite(r"mask.png", mask)
        import os
        # if a.is_dir():
        #     os.startfile(outPutPath)
        os.remove(outPutPath + 'back_up.png')
        print(outPutPath + "抠图结束")
        os.remove('frame_base.png')
        # cv2.waitKey(0)

    def play3(self):
        print("开始抠图" + self.outpath)
        outPutPath = self.outpath
        a = Path(outPutPath)
        a.mkdir(exist_ok=True)

        # 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
        imageA = cv2.imread(self.png1, cv2.IMREAD_UNCHANGED)
        imageB = cv2.imread(self.png2, cv2.IMREAD_UNCHANGED)
        imgB = Image.open(self.png2)
        imgB.save(outPutPath + 'back_up.png')
        imageC = cv2.imread(outPutPath + "back_up.png", cv2.IMREAD_UNCHANGED)
        # grayA = cv2.cvtColor(imageA, cv2.COLOR_RGBA2GRAY)
        # grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        self.wholeH, self.wholeW, self.channels = imageB.shape

        if self.wholeW<1600:
            self.minW = self.minW/2
            self.minH = self.minH/2
            self.minR = self.minR/2

        cnts, hierarchy = cv2.findContours(imageA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 找到一系列区域，在区域周围放置矩形
        for i in range(0, len(cnts)):
            area = cv2.contourArea(cnts[i])#轮廓的面积
            self.JudgmentContains(cnts[i])
        n = 1
        filename = outPutPath + "coordinate"
        f = open(filename, "w")
        f.write(f"{round(self.wholeW * self.px2cm, 3)},{round(self.wholeH * self.px2cm, 3)}" + '\n')
        self.index = 1
        for cnts in self.cntsList:
            if cnts is None:
                continue
            (x, y, w, h) = cv2.boundingRect(cnts)  # 得到外接矩形
            rect = cv2.minAreaRect(cnts)  # 得到面积最小的外接矩形
            (c_x, c_y), raidus = cv2.minEnclosingCircle(cnts)  # 得到外接圆
            curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
            cv2.imwrite(outPutPath + "mix_" + str(n) + ".png", curImage)
            self.get_location_file(x + w / 2, y + h / 2, w, h, f)

            points = cv2.boxPoints(rect)
            points = np.int0(points)

            center = (int(c_x), int(c_y))
            r = int(raidus)
            rectS = w * h
            rectC = raidus * raidus * math.pi

            oldw = w
            oldh = h
            oldr = r

            if r < self.minR:
                # print(r)
                r = self.minR
            centerCut = (r, r)
            if w < h:
                if w < self.minW: w = self.minW
                if h < self.minH: h = self.minH
            else:
                if h < self.minW: h = self.minW
                if w < self.minH: w = self.minH
            # if rectS > rectC:
            #     cv2.circle(imageB, center, r, (95,235,95), 10)  # 画外接圆
            #     # cv2.circle(imageB, center, oldr, (95, 235, 95), 10)  # 画老的，小的外接圆
            #     imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
            #     imageCircle[:] = (0, 0, 0, 0)
            #     cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 42)  # 画每个抠图的圆边框
            #     cv2.imwrite(outPutPath + "mix_" + str(n) + "_frame.png", imageCircle)
            #
            # else:
            #     cv2.rectangle(imageB, (x, y), (x + w, y + h), (95,235,95), 10)  # 画外接矩形
            #     self.draw_rect((w, h), outPutPath + "mix_" + str(n) + "_frame.png")

            cv2.drawContours(imageB, [points], 0, (95, 235, 95, 255), 10)
            # self.draw_rect((w, h), outPutPath + "mix_" + str(n) + "_frame.png")
            n += 1
        f.close()
        cv2.imwrite(outPutPath + "result.png", imageB)  # 得到原图的画框图

        import os
        os.remove(outPutPath + 'back_up.png')
        print(outPutPath + "抠图结束")
        os.remove('frame_base.png')


    def get_location_file(self, x, y, w, h, f):
        print(x, y, w, h, self.wholeW, self.wholeH)
        f.write(
            f"{self.index}:{round(x * self.px2cm, 3)},{round(y * self.px2cm, 3)}:{round(w * self.px2cm, 3)},{round(h * self.px2cm, 3)}" + '\n')
        self.index += 1

    # for line in content.splitlines():
    #     f.write(line + "\n")

    def draw_rect(self, size, save_path):

        thickness = 26

        imgbase = Image.open(self.baseframepng)
        base_width, base_height = imgbase.size
        img_new = Image.new('RGBA', size)
        new_width, new_height1 = size

        base_halfW = int(base_width / 2)
        base_halfH = int(base_height / 2)

        redW = new_width - base_width
        redH = new_height1 - base_height

        DW = new_width - base_halfW
        DH = new_height1 - base_halfH

        for x in range(0, base_halfW):
            for y in range(0, base_halfH):
                img_new.putpixel((x, y), imgbase.getpixel((x, y)))
        for x in range(base_halfW, base_width):
            for y in range(0, base_halfH):
                img_new.putpixel((x + redW, y), imgbase.getpixel((x, y)))
        for x in range(0, base_halfW):
            for y in range(base_halfH, base_height):
                img_new.putpixel((x, y + redH), imgbase.getpixel((x, y)))
        for x in range(base_halfW, base_width):
            for y in range(base_halfH, base_height):
                img_new.putpixel((x + redW, y + redH), imgbase.getpixel((x, y)))
        for x in range(base_halfW, DW):
            for y in range(0, thickness):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(base_halfW, DW):
            for y in range(new_height1 - thickness, new_height1):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(0, thickness):
            for y in range(base_halfH, DH):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(new_width - thickness, new_width):
            for y in range(base_halfH, DH):
                img_new.putpixel((x, y), (95, 235, 95, 255))

        img_new.save(save_path)

class AutoMattingPSD():
    def __init__(self, psd, outpath , is_save_huidu):
        self.psd = psd
        # self.baseframepng = "G:\\img_lib/frame_base.png"
        self.minW = 350
        self.minH = 380
        self.min_e_W = 350
        self.min_e_H = 500
        self.minR = 200
        self.outpath = outpath
        # self.px2cm = 1 / 28.35
        self.px2cm = 25.4 / 3000
        self.layer_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.is_save_mix_huidu_img = is_save_huidu

        from frame_base_png import img as frame_base
        tmp = open('frame_base.png', 'wb')
        tmp.write(base64.b64decode(frame_base))
        tmp.close()
        self.baseframepng = "frame_base.png"

    def play(self):
        print("开始处理：" + self.outpath)
        outPutPath = self.outpath
        a = Path(outPutPath)
        a.mkdir(exist_ok=True)

        psd = PSDImage.open(self.psd)
        img_num_base = None
        img_base = None

        for layer in psd:
            if layer.is_group():
                for baselayer in layer:
                    if baselayer.name == "base":
                        img_num_base = baselayer.numpy()
                        img_base = baselayer.composite()
                        # cv2.imshow("test1", img_num_base)
                        # cv2.waitKey(0)
                        # img_png = psd.composite()
                        # img_png.save('test_psd.png')
            elif layer.name == "base":
                img_num_base = layer.numpy()
                img_base = layer.composite()
                break
        if img_base is not None:
            img_num_base = np.array(img_base)  # imagebase转换成numpy格式
            gray_base = cv2.cvtColor(img_num_base, cv2.COLOR_RGBA2GRAY)
            img_png = psd.composite()


            img_num_all_mix = np.array(img_png)
            gray_all_mix = cv2.cvtColor(img_num_all_mix, cv2.COLOR_RGB2GRAY)
            # img_num_all_mix = img_num_base #在有不同点的图上画图
            (all_score, all_diff) = structural_similarity(gray_base, gray_all_mix, full=True)
            all_diff = (all_diff * 255).astype("uint8")
            all_thresh = cv2.threshold(all_diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # all_thresh1 = all_thresh.copy()  # 画椭圆的基础图

            base_h, base_w, baee_n = img_num_base.shape

            filename = f"{outPutPath}coordinate"
            f = open(filename, "w")
            f.write(f"{round(base_w * self.px2cm, 3)},{round(base_h * self.px2cm, 3)}" + '\n')


            # 保存base图片
            resize_img_png = img_png
            base_size_w, base_size_h = resize_img_png.size
            base_size_w, base_size_h = base_size_w / 2, base_size_h / 2
            resize_img_png.thumbnail((base_size_w, base_size_h), resample=Image.LANCZOS)  # 对base图片进行缩放
            resize_img_png.save(f'{outPutPath}base.png')


            for curlayer in reversed(psd):
                if curlayer.is_group():
                    continue
                elif curlayer.name in self.layer_list:
                    img_cur_mix = curlayer.composite()
                    cur_size_w, cur_size_h = img_cur_mix.size
                    cur_size_w, cur_size_h = cur_size_w / 2, cur_size_h / 2
                    img_cur_mix.thumbnail((cur_size_w, cur_size_h), resample=Image.LANCZOS)
                    img_cur_mix.save(f"{outPutPath}mix_{curlayer.name}.png")

                    img_mix = psd.composite(layer_filter=lambda
                        layer: layer.name == curlayer.name or layer.name == 'base')  # or layer.name == '图层 0')
                    img_num_mix = np.array(img_mix)
                    # img_base.save('base_psd.png')
                    # img_mix.save('mix_psd.png')
                    gray_mix = cv2.cvtColor(img_num_mix, cv2.COLOR_RGB2GRAY)
                    (score, diff) = structural_similarity(gray_base, gray_mix, full=True)
                    diff = (diff * 255).astype("uint8")
                    # print(type(diff.shape))

                    # fill_all_pixels(diff,x,y,weight,height)
                    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                    point_list = []
                    for curcnt in cnts:
                        for point in curcnt:
                            point_list.append(point)

                    cnt = np.array(point_list)



                    (r_x, r_y, r_w, r_h) = cv2.boundingRect(cnt)  # 普通外接矩形轮廓
                    if r_w < r_h:
                        if r_w < self.minW:
                            d_x = self.minW - r_w
                            r_x -= d_x / 2
                            r_w = self.minW
                        if r_h < self.minH:
                            d_y = self.minH - r_h
                            r_y -= d_y / 2
                            r_h = self.minH
                    else:
                        if r_h < self.minW:
                            d_y = self.minW - r_h
                            r_h = self.minW
                            r_y -= d_y / 2
                        if r_w < self.minH:
                            d_x = self.minH - r_w
                            r_x -= d_x / 2
                            r_w = self.minH
                    if r_x < 0: r_x = 0
                    if r_y < 0: r_y = 0
                    if r_x + r_w > base_w: r_x = base_w - r_w
                    if r_y + r_h > base_h: r_y = base_h - r_h
                    r_x = int(r_x)
                    r_y = int(r_y)

                    rect = cv2.minAreaRect(cnt)  # 最小矩形轮廓
                    (r_center, (weight, height), angle) = rect
                    old_weight, old_height = weight, height

                    (e_x, e_y), (e_a, e_b), e_angle = cv2.fitEllipse(
                        cnt)  # 椭圆轮廓 x, y）代表椭圆中心点的位置（a, b）代表长短轴长度，应注意a、b为长短轴的直径，而非半径 angle 代表了中心旋转的角度
                    e_a = 1.15 * e_a  # 短轴
                    e_b = 1.15 * e_b  # 长轴

                    if weight < height:
                        if weight < self.minW: weight = self.minW
                        if height < self.minH: height = self.minH
                    else:
                        if height < self.minW: height = self.minW
                        if weight < self.minH: weight = self.minH
                    if e_a < self.min_e_W: e_a = self.min_e_W
                    if e_b < self.min_e_H: e_b = self.min_e_H

                    sR = weight * height
                    r_c_x , r_c_y = r_center
                    rect = (r_center, (weight, height), angle)
                    points = cv2.boxPoints(rect)
                    points = np.intp(points)  # 最小矩形轮廓

                    retval = (e_x, e_y), (e_a, e_b), e_angle  # 椭圆轮廓
                    sE = np.pi * e_a / 2 * e_b / 2  # 椭圆面积

                    (c_x, c_y), radius = cv2.minEnclosingCircle(cnt)  # 圆轮廓
                    center = (int(c_x), int(c_y))
                    r = int(radius)
                    oldr = r
                    if r < self.minR:
                        r = self.minR
                    centerCut = (r, r)
                    sC = r * r * math.pi

                    out_path = f"{outPutPath}mix_{curlayer.name}_frame.png"
                    f_c_x = 0
                    f_c_y = 0

                    if sE < sR and sE < sC:
                        is_draw_min_ellipse = True
                        focus = self.get_focus((e_x, e_y), (e_a / 2, e_b / 2), e_angle)
                        p1, p2 = np.intp(focus)

                        #重要参数打印出来
                        # print(curlayer.name, (e_x, e_y), (e_a, e_b), e_angle, focus)

                        for i in range(base_w):
                            if not is_draw_min_ellipse:
                                break
                            pos0 = (i, 0)
                            pos1 = (i, 2000)
                            if self.is_in_elliptic(pos0, focus, e_b) or self.is_in_elliptic(pos1, focus, e_b):
                                is_draw_min_ellipse = False
                        for i in range(base_h):
                            if not is_draw_min_ellipse:
                                break
                            pos0 = (0, i)
                            pos1 = (3000, i)
                            if self.is_in_elliptic(pos0, focus, e_b) or self.is_in_elliptic(pos1, focus, e_b):
                                is_draw_min_ellipse = False
                        if is_draw_min_ellipse:
                            # cv2.line(img_num_all_mix,p1,p2,(95, 235, 95, 255), 10) #画出椭圆焦点连起来的线段
                            cv2.ellipse(all_thresh, retval, (95, 235, 95, 255), 26)  # 椭圆
                            cv2.ellipse(img_num_all_mix, retval, (95, 235, 95, 255), 26)  # 椭圆
                            self.draw_elliptic((int(e_a), int(e_b)), 180 - int(e_angle), out_path)  # 切出椭圆
                            # print(curlayer.name,(e_x, e_y), (e_a, e_b), e_angle)
                            f_c_x = e_x
                            f_c_y = e_y
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95),
                                          10)  # 画外接矩形
                            size = (int(r_w), int(r_h))
                            self.draw_rect(size, 0, out_path)  # 切出矩形
                            f_c_x = r_x + r_w / 2
                            f_c_y = r_y + r_h / 2
                    elif sC <= sE and sC <= sR:
                        # print(curlayer.name,'画圆')
                        is_draw_min_circle = True
                        for i in range(base_w):
                            if not is_draw_min_circle:
                                break
                            pos0 = (i, 0)
                            pos1 = (i, 2000)
                            if self.is_in_circle(pos0, center, r) or self.is_in_circle(pos1, center, r):
                                is_draw_min_circle = False
                        for i in range(base_h):
                            if not is_draw_min_circle:
                                break
                            pos0 = (0, i)
                            pos1 = (3000, i)
                            if self.is_in_circle(pos0, center, r) or self.is_in_circle(pos1, center, r):
                                is_draw_min_circle = False
                        if is_draw_min_circle:
                            cv2.circle(all_thresh, center, r, (95, 235, 95), 26)  # 画外接圆
                            cv2.circle(img_num_all_mix, center, r, (95, 235, 95), 26)  # 画外接圆
                            imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                            imageCircle[:] = (0, 0, 0, 0)
                            cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 26)  # 画每个抠图的圆边框
                            height, width = imageCircle.shape[:2]
                            resize = (int(width / 2), int(height / 2))
                            resize_img = cv2.resize(imageCircle, resize, interpolation=cv2.INTER_AREA)
                            cv2.imwrite(out_path, resize_img)
                            f_c_x, f_c_y = center
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95),
                                          10)  # 画外接矩形
                            size = (int(r_w), int(r_h))
                            self.draw_rect(size, 0, out_path)  # 切出矩形
                            f_c_x = r_x + r_w / 2
                            f_c_y = r_y + r_h / 2
                    else:
                        # print(type([points][0]))
                        is_draw_min_rect = True
                        for pos in points:
                            if pos[0] < 0 or pos[0] > 3000:
                                is_draw_min_rect = False
                                break
                            if pos[1] < 0 or pos[1] > 2000:
                                is_draw_min_rect = False
                                break
                        if is_draw_min_rect:
                            cv2.drawContours(all_thresh, [points], 0, (95, 235, 95, 255), 26)  # 最小矩形
                            cv2.drawContours(img_num_all_mix, [points], 0, (95, 235, 95, 255), 26)  # 最小矩形
                            size = (int(weight), int(height))
                            self.draw_rect(size, 180 - int(angle), out_path)  # 切出矩形
                            f_c_x = r_c_x + weight / 2
                            f_c_y = r_c_y + height / 2
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95),
                                          10)  # 画外接矩形
                            size = (int(r_w), int(r_h))
                            self.draw_rect(size, 0, out_path)  # 切出矩形
                            f_c_x = r_x + r_w / 2
                            f_c_y = r_y + r_h / 2

                    lt_x, lt_y = curlayer.offset
                    c_w, c_h = curlayer.size
                    c_x, c_y = lt_x + c_w / 2, lt_y + c_h / 2
                    f.write(
                        f"{curlayer.name}:{round(c_x * self.px2cm, 3)},{round(c_y * self.px2cm, 3)}:{round(f_c_x * self.px2cm, 3)},{round(f_c_y * self.px2cm, 3)}" + '\n')
                    print(f'不同点 {curlayer.name} 处理完毕')
                    # cv2.circle(img_num_all_mix, center, r, (95, 235, 95, 255), 10)  # 圆
                    # cv2.ellipse(img_num_all_mix, retval, (95, 235, 95, 255), 10)  # 椭圆
                    # cv2.drawContours(img_num_all_mix, [points], 0, (95, 235, 95, 255), 10)  # 矩形

                    # points_list = [points]
                    # max_x = points_list[0][0][0]
                    # max_y = points_list[0][0][1]
                    # min_x = points_list[0][0][0]
                    # min_y = points_list[0][0][1]
                    # for point in points:
                    #     if point[0] > max_x:
                    #         max_x = point[0]
                    #     elif point[0] < min_x:
                    #         min_x = point[0]
                    #     if point[1] > max_y:
                    #         max_y = point[1]
                    #     elif point[1] < min_y:
                    #         min_y = point[1]
                    # size = (w,h) = (max_x-min_x,max_y-min_y)

                    # retval = cv2.moments(thresh.copy())
                    # print(len(cnts))
            if(self.is_save_mix_huidu_img):
                cv2.imwrite(f"{outPutPath}huidu.png", all_thresh)
                cv2.imwrite(f"{outPutPath}/mix.png", img_num_all_mix)
            print(outPutPath + '处理完毕')
        else:
            print(outPutPath + 'base图不存在或没找到')
    def rotate_image(self,input_image, angle):
        # input_rows, input_cols, channels = input_image.shape
        # assert channels == 3

        # 1. Create an output image with the same shape as the input
        # output_image = np.zeros_like(input_image)
        # #获取中点坐标，具体参照np.shape()
        # x0 = input_image.shape[0]/2
        # y0 = input_image.shape[1]/2
        #
        # w = input_image.shape[0]
        # h = input_image.shape[1]
        #
        # print(x0,y0)
        # for i in range(w):#width
        #     for j in range(h):#heigh
        #     	# """
        #         # (i,j)旋转后的坐标为(x,y),公式 数学推导
        #         # 一点(x,y)绕点(x0,y0)旋转θ角度后的坐标(x`,y`)为：
        #         # { x` = (x - x0)*cos - (y - y0)*sin + x0
        # 		# { y` = (x - x0)*sin + (y - y0)*cos + y0
        # 		# """
        #         x = (int)((i - x0)*np.cos(theta) - (j - y0)*np.sin(theta) + x0)
        #         y = (int)((i - x0)*np.sin(theta) + (j - y0)*np.cos(theta) + y0)
        #         if x < w and y < h and x >= 0 and y >= 0:#满足的所有条件
        #             output_image[i][j] = input_image[x][y]
        # cv2.imwrite(f"{outPutPath}test.png", output_image)
        rows, cols = input_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        # 自适应图片边框大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = rows * sin + cols * cos
        new_h = rows * cos + cols * sin
        M[0, 2] += (new_w - cols) * 0.5
        M[1, 2] += (new_h - rows) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
        res2 = cv2.warpAffine(input_image, M, (w, h))
        return res2
        # cv2.imwrite(f"{outPutPath}test.png", res2)
        # 画图部分
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # # plt.title('原图')
        # plt.axis(False)
        # plt.subplot(122)
        # plt.imshow(res2)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # # plt.title('绕中心点旋转的图像')
        # plt.axis(False)
        # plt.show()

        # return output_image

    def draw_rect(self,size, angle, save_path):
        thickness = 26

        imgbase = Image.open(self.baseframepng)
        base_width, base_height = imgbase.size
        img_new = Image.new('RGBA', size)
        new_width, new_height1 = size

        base_halfW = int(base_width / 2)
        base_halfH = int(base_height / 2)

        redW = new_width - base_width
        redH = new_height1 - base_height

        DW = new_width - base_halfW
        DH = new_height1 - base_halfH

        for x in range(0, base_halfW):
            for y in range(0, base_halfH):
                img_new.putpixel((x, y), imgbase.getpixel((x, y)))
        for x in range(base_halfW, base_width):
            for y in range(0, base_halfH):
                img_new.putpixel((x + redW, y), imgbase.getpixel((x, y)))
        for x in range(0, base_halfW):
            for y in range(base_halfH, base_height):
                img_new.putpixel((x, y + redH), imgbase.getpixel((x, y)))
        for x in range(base_halfW, base_width):
            for y in range(base_halfH, base_height):
                img_new.putpixel((x + redW, y + redH), imgbase.getpixel((x, y)))
        for x in range(base_halfW, DW):
            for y in range(0, thickness):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(base_halfW, DW):
            for y in range(new_height1 - thickness, new_height1):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(0, thickness):
            for y in range(base_halfH, DH):
                img_new.putpixel((x, y), (95, 235, 95, 255))
        for x in range(new_width - thickness, new_width):
            for y in range(base_halfH, DH):
                img_new.putpixel((x, y), (95, 235, 95, 255))

        cur_size_w, cur_size_h = img_new.size
        cur_size_w, cur_size_h = cur_size_w / 2, cur_size_h / 2
        img_new.thumbnail((cur_size_w, cur_size_h), resample=Image.LANCZOS)
        img = np.array(img_new)
        cv2.imwrite(save_path, self.rotate_image(img, angle))
        # img_new.save(save_path)

    def draw_elliptic(self,size, angle, save_path):
        thickness = 26
        # print(size , angle ,save_path)
        new_size = (size[0] + 26, size[1] + 26)
        img_new = Image.new('RGBA', new_size)
        img = np.array(img_new)
        new_width, new_height = size
        x, y = new_size
        # shape = (size,4)
        # img = numpy.zeros(shape, dtype=float, order='C')
        cv2.ellipse(img, ((x / 2, y / 2), (new_width, new_height), 0), (95, 235, 95, 255), thickness)  # 椭圆

        height,width = img.shape[:2]
        resize = (int(width/2),int(height/2))
        resize_img = cv2.resize(img,resize,interpolation = cv2.INTER_AREA)
        cv2.imwrite(save_path, self.rotate_image(resize_img, angle))

    def get_focus(self,center_pos, size, angle):  # 获得椭圆焦点
        short, long = size
        center_x, center_y = center_pos
        focus_len = math.pow(long * long - short * short, 0.5)
        ang = 180 - angle
        # print(math.cos(math.radians(ang)),math.sin(math.radians(-ang)))
        radians1 = math.radians(-ang)
        radians2 = math.radians(ang)
        x0 = focus_len * math.sin(radians1)
        y0 = focus_len * math.cos(radians2)  # 以原点为中心点的时候
        # print("x0,y0",x0,y0,focus_len,angle)
        y1 = -y0 + center_y
        x1 = x0 + center_x
        y2 = y0 + center_y
        x2 = -x0 + center_x

        return [(x1, y1), (x2, y2)]

    def is_in_elliptic(self,point, focus, long):
        focus1, focus2 = focus
        len1 = self.get_len(point, focus1)
        len2 = self.get_len(point, focus2)
        if len1 + len2 <= long:
            return True
        return False

    def is_in_circle(self,point, center, r):
        len = self.get_len(point, center)
        if len <= r:
            return True
        return False

    def get_len(self,pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        dx = x1 - x2
        dy = y1 - y2
        return math.pow(dx * dx + dy * dy, 0.5)










def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if int(radius) > 1:
        radius = 1

    corner_radius = int(radius * (height / 2))

    if thickness < 0:
        # big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90,
                color, thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90,
                color, thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,
                color, thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,
                color, thickness, line_type)

    return src


# top_left = (0, 0)
# bottom_right = (500, 800)
# color = (255, 0, 0)
# image_size = (500, 800, 3)
# img = np.zeros(image_size)
# img = rounded_rectangle(img, top_left, bottom_right, color=color, radius=0.5, thickness=-1)
#
# cv2.imshow('rounded_rect', img)
# cv2.waitKey(0)


# cntsList = []
# png1 = "111.png"
# png2 = "222.png"
# minArea = 2000
# def JudgmentContains(curCnts):
#     for index in range(len(cntsList)):
#         (x, y, w, h) = cv2.boundingRect(cntsList[index])
#         (x1, y1, w1, h1) = cv2.boundingRect(curCnts)
#         if x1 >= x and y1 >= y and x1 + w1 <= x + w and y1 + h1 <= y + h:  # 传入的框被之前的框包含
#             return
#             print(11111)
#         elif x1 <= x and y1 <= y and x1 + w1 >= x + w and y1 + h1 >= y + h:  # 传入的框包含之前的框
#             cntsList[index] = None
#             if not JudgmentList(curCnts):
#                 cntsList.append(curCnts)
#             print(2222)
#         elif (x1 + w1 >= x and x + w >= x1 and y1 + h1 >= y and y + h >= y1):  # 传入的框相交之前的框
#             print(4444)
#             s = w * h
#             s1 = w1 * h1
#             if s1 > s:
#                 cntsList[index] = None
#                 if not JudgmentList(curCnts):
#                     cntsList.append(curCnts)
#                 print(3333)
#             else:
#                 return
#     if not JudgmentList(curCnts):
#         cntsList.append(curCnts)
#
#
# def JudgmentList(curCnts):
#     (x, y, w, h) = cv2.boundingRect(curCnts)
#     for cnt in cntsList:
#         (x1, y1, w1, h1) = cv2.boundingRect(cnt)
#         if (x, y, w, h) == (x1, y1, w1, h1):
#             return True
#     return False
#
#
# # 加载两张图片并将他们转换为灰度 IMREAD_GRAYSCALE IMREAD_UNCHANGED COLOR_RGB2GRAY
# imageA = cv2.imread(png1, cv2.IMREAD_UNCHANGED)
# imageB = cv2.imread(png2, cv2.IMREAD_UNCHANGED)
# imgB = Image.open(png2)
# imgB.save('back_up.png')
# imageC = cv2.imread("back_up.png", cv2.IMREAD_UNCHANGED)
# grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
#
# # 计算两个灰度图像之间的结构相似度指数
# (score, diff) = structural_similarity(grayA, grayB, full=True)
# diff = (diff * 255).astype("uint8")
# # print("SSIM:{}".format(score))
#
# # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
# thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imwrite(r"huidu.png", thresh)
# cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if imutils.is_cv4() else cnts[1]
# # print(len(cnts),'\n',hierarchy,type(hierarchy))
#
# # mask = np.zeros(imageA.shape, dtype='uint8')
#
# # 找到一系列区域，在区域周围放置矩形
# for i in range(0, len(cnts)):
#     # if not hierarchy[i][3] == -1:
#     #     continue
#     area = cv2.contourArea(cnts[i])
#     # cv2.arcLength() 轮廓的周长
#     if area < minArea:
#         continue
#     JudgmentContains(cnts[i])
#
#     # (x, y, w, h) = cv2.boundingRect(cnts[i])
#     # curImage = cv2.getRectSubPix(imageC,(w,h),(x + w/2,y + h/2))
#     # width, height = curImage.size
#     # for i in range(0,width):
#     #     for j in range(0,height):
#     #         # color = curImage.getpixel((i, j))
#     #         if curImage.pointPolygonTest(cnts[i],(i,j),True) < 0:
#     #             curImage.putpixel((i, j), (0, 0, 0,0))
#     # cv2.imwrite("./koutu/"+str(n)+".png", curImage)
#     # n+=1
#     # cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 10)
#     # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 10)
#     # cv2.drawContours(imageB, cnts[i], -1, (0, 255, 0), 10) #可以画出轮廓的线
#     # center = (int(x),int(y))
#     # (x,y),raidus = cv2.minEnclosingCircle(c) #得到外接圆
#     # cv2.circle(imageB, center, int(raidus), (0, 255, 0), 10)# 画到外接圆
#     # cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
#     # 可以绘制斜着的框
#     # rect = cv2.minAreaRect(cnts[i])
#     # box = cv2.boxPoints(rect)
#     # box = np.int0(box)
#     # cv2.drawContours(imageB, [box], 0, (0, 255, 0), 10)
# n = 1
# if not os.path.exists("./koutu/"):
#     os.mkdir("./koutu/")
# for cnts in cntsList:
#     if cnts is None:
#         continue
#     (x, y, w, h) = cv2.boundingRect(cnts)  # 得到外接矩形
#     (c_x, c_y), raidus = cv2.minEnclosingCircle(cnts)  # 得到外接圆
#     curImage = cv2.getRectSubPix(imageC, (w, h), (x + w / 2, y + h / 2))
#     cv2.imwrite("./koutu/" + str(n) + ".png", curImage)
#
#     center = (int(c_x), int(c_y))
#     r = int(raidus)
#     centerCut = (r, r)
#     rectS = w * h
#     rectC = raidus * raidus * math.pi
#     if rectS > rectC:
#         cv2.circle(imageB, center, r, (0, 255, 0), 10)  # 画外接圆
#         imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
#         imageCircle[:] = (0, 0, 0, 0)
#         cv2.circle(imageCircle, centerCut, r - 5, (0, 255, 0, 255), 10)  # 画每个抠图的圆边框
#         cv2.imwrite("./koutu/" + str(n) + "_rect.png", imageCircle)
#     else:
#         cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 10)  # 画外接矩形
#         imageRect = np.zeros((curImage.shape[0], curImage.shape[1], 4))  # 创建opencv图像
#         imageRect[:] = (0, 0, 0, 0)
#         cv2.rectangle(imageRect, (5, 5), (w - 5, h - 5), (0, 255, 0, 255), 10)  # 画每个抠图的边框
#         cv2.imwrite("./koutu/" + str(n) + "_rect.png", imageRect)
#     # print(curImage.size)
#     n += 1
#
# # imageRect = np.zeros((500,500,4))
# # imageRect[:] = (0, 0, 0, 0)
# # cv2.imshow("Modified", imageRect)
# # 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
# # //cv2.imshow("Modified", imageB)
# # cv2.imwrite(r"mask.png", mask)
# cv2.imwrite("result.png", imageB)
# print("end")
# # cv2.waitKey(0)
import numpy
