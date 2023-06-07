import base64
import PIL
from psd_tools import PSDImage
from PIL import Image
import math
import operator
from functools import reduce

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

class AutoMattingPSD():
    def __init__(self, psd, outpath , is_save_huidu , psd_name):
        self.psd = psd
        # self.baseframepng = "G:\\img_lib/frame_base.png"
        self.minW = 350
        self.minH = 380
        self.min_e_W = 350
        self.min_e_H = 500
        self.minR = 200
        self.outpath = outpath
        self.mix_outpath = f"{outpath}mix/"
        self.psd_name = psd_name
        # self.px2cm = 1 / 28.35
        self.px2cm = 25.4 / 3000
        self.layer_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.is_save_mix_huidu_img = is_save_huidu
        self.green_color = (95, 235, 95, 255)
        self.color = (95, 235, 95)

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
        b = Path(self.mix_outpath)
        b.mkdir(exist_ok=True)

        psd = PSDImage.open(self.psd)
        psd.save(f"{outPutPath}{self.psd_name}")
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
            resize_img_base_png = img_base
            base_size_w, base_size_h = resize_img_base_png.size
            base_size_w, base_size_h = base_size_w / 2, base_size_h / 2
            resize_img_base_png.thumbnail((base_size_w, base_size_h), resample=Image.LANCZOS)  # 对base图片进行缩放
            resize_img_base_png.save(f'{outPutPath}base.png')


            for curlayer in reversed(psd):
                if curlayer.is_group():
                    continue
                elif curlayer.name in self.layer_list:
                    img_cur_mix = curlayer.composite()
                    cur_size_w, cur_size_h = img_cur_mix.size
                    cur_size_w, cur_size_h = cur_size_w / 2, cur_size_h / 2
                    img_cur_mix.thumbnail((cur_size_w, cur_size_h), resample=Image.LANCZOS)
                    img_cur_mix.save(f"{self.mix_outpath}mix_{curlayer.name}.png")

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
                    e_a = 1.19 * e_a  # 短轴
                    e_b = 1.19 * e_b  # 长轴

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
                    centerCut = (r + 14, r + 14)
                    sC = r * r * math.pi

                    out_path = f"{self.mix_outpath}mix_{curlayer.name}_frame.png"
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
                            cv2.ellipse(img_num_all_mix, retval, (255, 0, 0, 255), 26)  # 椭圆
                            self.draw_elliptic((int(e_a), int(e_b)), 180 - int(e_angle), out_path)  # 切出椭圆
                            # print(curlayer.name,(e_x, e_y), (e_a, e_b), e_angle)
                            f_c_x = e_x
                            f_c_y = e_y
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 26)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (127, 0, 0),26)  # 画外接矩形
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
                            cv2.circle(img_num_all_mix, center, r, (0, 255, 0), 26)  # 画外接圆
                            self.draw_circle(r,centerCut,out_path)
                            # imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                            # imageCircle[:] = (0, 0, 0, 0)
                            # cv2.circle(imageCircle, centerCut, r - 13, (95, 235, 95, 255), 26)  # 画每个抠图的圆边框
                            # height, width = imageCircle.shape[:2]
                            # resize = (int(width / 2), int(height / 2))
                            # resize_img = cv2.resize(imageCircle, resize, interpolation=cv2.INTER_AREA)
                            # cv2.imwrite(out_path, resize_img)
                            f_c_x, f_c_y = center
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 26)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (0, 127, 0),26)  # 画外接矩形
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
                            cv2.drawContours(img_num_all_mix, [points], 0, (0, 0, 255, 255), 26)  # 最小矩形
                            size = (int(weight), int(height))
                            self.draw_rect(size, 180 - int(angle), out_path)  # 切出矩形
                            f_c_x = r_c_x# + weight / 2
                            f_c_y = r_c_y# + height / 2
                        else:
                            cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 26)  # 画普通外接矩形
                            cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (0, 0, 255),26)  # 画外接矩形
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
        cv2.ellipse(img, ((x / 2, y / 2), (new_width, new_height), 0), (95, 235, 95, 255), thickness,cv2.LINE_AA)  # 椭圆

        height,width = img.shape[:2]
        resize = (int(width/2),int(height/2))
        resize_img = cv2.resize(img,resize,interpolation = cv2.INTER_LINEAR) #INTER_NEARAST INTER_LINEAR INTER_AREA INTER_CUBIC INTER_LANCZOS4
        rotate_npary_img = self.rotate_image(resize_img, angle)
        rotate_pil_img = PIL.Image.fromarray(rotate_npary_img)
        w,h = rotate_pil_img.size
        r,g,b = self.color
        for wi in range(0,w):
            for he in range(0,h):
                pix_color = rotate_pil_img.getpixel((wi,he))
                # r, g, b,a = pix_color
                alpha = pix_color[3]
                if not (pix_color == self.green_color or alpha == 0):
                    # if pix_color[3] > 0:
                    rotate_pil_img.putpixel((wi,he),(r,g,b,alpha))#消除锯齿
                    # rotate_pil_img.putpixel((wi,he),(r,g,b,255))
                    # else:
                    #     rotate_pil_img.putpixel((wi,he),(0,0,0,0))
        rotate_pil_img.save(save_path)
        # cv2.imwrite(save_path, rotate_npary_img)

    def draw_circle(self, r, centerCut, out_path):
        # imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
        # imageCircle[:] = (0, 0, 0, 0)
        print("draw_circle")
        new_size = (r*2 +28, r*2 +28)
        img_new = Image.new('RGBA', new_size)
        img = np.array(img_new)

        cv2.circle(img, centerCut, r , (95, 235, 95, 255), 26, cv2.LINE_AA)  # 画每个抠图的圆边框
        height, width = img.shape[:2]
        resize = (int(width / 2), int(height / 2))
        resize_img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(out_path, resize_img)
        pil_img = PIL.Image.fromarray(resize_img)
        w, h = pil_img.size
        r,g,b = self.color
        for wi in range(0, w):
            for he in range(0, h):
                pix_color = pil_img.getpixel((wi, he))
                alpha = pix_color[3]
                if not (pix_color == self.green_color or alpha == 0):
                    pil_img.putpixel((wi,he),(r,g,b,alpha))#消除锯齿

                    # if pix_color[3] > 0:
                    #     pil_img.putpixel((wi, he), self.green_color)
                    # else:
                    #     pil_img.putpixel((wi, he), (0, 0, 0, 0))
        pil_img.save(out_path)

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