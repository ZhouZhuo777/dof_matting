import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mes(image1, image2):  # 均方误差（MSE）
    mes_value = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    mes_value /= float(image1.shape[0] * image1.shape[1])
    return mes_value


# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')


def ssim(image1, image2):  # 结构相似性度量（calculateSSIM）
    ssim_value = cv2.SSIM(image1, image2)
    return ssim_value


# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

def ncc(image1, image2):  # 归一化交叉相关（NCC）
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    cov = np.sum(image1 - mean1) * (image2 - mean2)
    sd1 = np.sum((image1 - mean1) ** 2)
    sd2 = np.sum((image2 - mean2) ** 2)
    return cov / np.sqrt(sd1 * sd2)


# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

def hamming_distance(image1, image2):  # 汉明窗口（Hamming distance)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift1 = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift1.detectAndCompute(gray1, None)
    sift2 = cv2.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = sift2.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    distance = 0
    for match in matches:
        distance += match.distance
    return distance / len(matches)


# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

# 三直方图算法相似度 单通道
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree + degree + 1
    degree = degree / len(hist1)
    return degree


# 三直方图算法相似度 三通道
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# imagee1_path = 'koutu/result.png'
# imagee2_path = 'koutu/result1.png'
# imagee3_path = 'koutu/333.png'
# imagee4_path = 'koutu/444.png'
# image1 = cv2.imread(imagee1_path)
# image2 = cv2.imread(imagee2_path)
# image3 = cv2.imread(imagee3_path)
# image4 = cv2.imread(imagee4_path)
#
# # print(hamming_distance(image1, image2))
# print(calculate(image1, image2))
# print(classify_hist_with_split(image1, image2))
# # print(hamming_distance(image1, image2))
# # print(ncc(image1, image2))
# # print(ssim(image1, image2))
# print(mes(image1, image2))
# print('\n')
#
# print(calculate(image1, image3))
# print(classify_hist_with_split(image1, image3))
# # print(hamming_distance(image1, image3))
# # print(ncc(image1, image3))
# # print(ssim(image1, image3))
# # print(mes(image1, image3))
# print('\n')
#
# print(calculate(image1, image4))
# print(classify_hist_with_split(image1, image4))
# # print(hamming_distance(image1, image4))
# # print(ncc(image1, image4))
# # print(ssim(image1, image4))
# # print(mes(image1, image4))
# print('\n')

# print(calculate(image3, image4))
# print(classify_hist_with_split(image3, image4))
# # print(hamming_distance(image3, image4))
# # print(ncc(image3, image4))
# # print(ssim(image3, image4))
# print(mes(image3, image4))
# print('\n')

# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc


###########################################################################
## Class ImageCompare
###########################################################################

class ImageCompare(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"image_compare", pos=wx.DefaultPosition,
                          size=wx.Size(500, 231), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer2 = wx.BoxSizer(wx.VERTICAL)

        sbSizer3 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, wx.EmptyString), wx.VERTICAL)

        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString), wx.HORIZONTAL)

        sbSizer2.Add((0, 0), 1, wx.EXPAND, 5)

        self.m_staticText1 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"文件夹路径", wx.Point(-1, -1),
                                           wx.DefaultSize, 0)
        self.m_staticText1.Wrap(-1)

        sbSizer2.Add(self.m_staticText1, 0, wx.ALL, 5)

        self.m_textCtrl1 = wx.TextCtrl(sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.Point(-1, -1),
                                       wx.Size(150, -1), 0)
        sbSizer2.Add(self.m_textCtrl1, 0, wx.ALL, 5)

        sbSizer2.Add((0, 0), 1, wx.EXPAND, 5)

        sbSizer3.Add(sbSizer2, 1, wx.EXPAND, 5)

        bSizer2.Add(sbSizer3, 1, wx.EXPAND, 5)

        bSizer4 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer4.Add((0, 0), 1, wx.EXPAND, 5)

        self.m_button5 = wx.Button(self, wx.ID_ANY, u"OK", wx.Point(-1, -1), wx.Size(120, 100), 0)

        self.m_button5.SetDefault()
        self.m_button5.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        self.m_button5.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE))

        bSizer4.Add(self.m_button5, 0, wx.ALL, 5)

        bSizer4.Add((0, 0), 1, wx.EXPAND, 5)

        bSizer2.Add(bSizer4, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer2)
        self.Layout()

        self.Centre(wx.BOTH)

    def __del__(self):
        pass


import wx
from pathlib import Path
import fnmatch
import os


class CompareImage:
    def __init__(self, image_list, folder):
        self.image_list = image_list
        self.result_list = []
        self.folder = folder
        self.have_find_image = []

    def compare(self):
        len_image = len(self.image_list)
        for i in range(len_image):
            cur_image_path = self.image_list[i]
            if cur_image_path in self.have_find_image:
                continue
            print(f"开始搜寻:{cur_image_path}")
            cur_image = cv2.imread(rf"{cur_image_path}")
            # if cur_image == None:
            #     raise Exception(f"图片解析出来为空  {cur_image_path}")
            max_similarity = 0
            max_similarity_image_path = None
            for j in range(len_image):
                mate_image_path = self.image_list[j]
                mate_image = cv2.imread(rf"{mate_image_path}")
                # if mate_image == None:
                #     raise Exception(f"图片解析出来为空  {mate_image_path}")
                cur_similarity = self.calculate(cur_image, mate_image)
                if cur_similarity > max_similarity:
                    max_similarity = cur_similarity
                    max_similarity_image_path = mate_image_path
            if max_similarity_image_path == None:
                print(f"{cur_image_path} 没有找到相似的图片")
            else:
                print(f"{cur_image_path} 与 {max_similarity_image_path}相似")
                self.have_find_image.append(cur_image_path)
                self.have_find_image.append(max_similarity_image_path)
                self.result_list.append((cur_image_path, max_similarity_image_path))
        self.rename()
        print("程序结束")

    def rename(self):
        i = 0
        for cur_tuple in self.result_list:
            i += 1
            os.rename(cur_tuple[0], self.folder + f"/{i}(1).png")
            os.rename(cur_tuple[1], self.folder + f"/{i}(2).png")
        print(f"改名完毕，总共{i}组图片")

    def calculate(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # plt.plot(hist1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree + degree + 1
        degree = degree / len(hist1)
        return degree


class image_compare(ImageCompare):
    def __init__(self, parent):
        super(image_compare, self).__init__(parent)
        self.m_button5.Bind(wx.EVT_BUTTON, self.on_button_clicked)

    def on_button_clicked(self, event):
        img_lib_path = self.m_textCtrl1.GetValue()
        libpath = Path(img_lib_path)

        if not libpath.is_dir():
            raise Exception("输入的路径不是一个文件夹")
        elif not libpath.exists():
            raise Exception("文件夹不存在")

        image_list = self.getAllFile(img_lib_path, '*.png')
        print('png数量：', len(image_list))
        if len(image_list) % 2 != 0:
            raise Exception("该文件夹的图片数量是奇数")
        png_path_list = []
        for cur_image in image_list:
            png_path = f"{img_lib_path}/{cur_image}"
            png_path_list.append(rf"{png_path}")

        compare_image = CompareImage(image_list=png_path_list, folder=img_lib_path)
        compare_image.compare()

    def getAllFile(self, path, key):
        res = []
        allfile = os.listdir(path)
        for filename in allfile:
            if fnmatch.fnmatch(filename, key):
                res.append(filename)
        return res


app = wx.App()
window = image_compare(parent=None)
window.Show()
app.MainLoop()
