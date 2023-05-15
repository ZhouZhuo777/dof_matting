import base64


def draw_rect(size, save_path):
    imgbase = Image.open("test.png")
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
        for y in range(0, 42):
            img_new.putpixel((x, y), (95, 235, 95, 255))
    for x in range(base_halfW, DW):
        for y in range(new_height1 - 42, new_height1):
            img_new.putpixel((x, y), (95, 235, 95, 255))
    for x in range(0, 42):
        for y in range(base_halfH, DH):
            img_new.putpixel((x, y), (95, 235, 95, 255))
    for x in range(new_width - 42, new_width):
        for y in range(base_halfH, DH):
            img_new.putpixel((x, y), (95, 235, 95, 255))

    img_new.save(save_path)


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
