import math

import PIL.Image as Image
import numpy
import skimage
from psd_tools import PSDImage
import imutils
import cv2
import os
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

def fill_all_pixels(diff: numpy.ndarray,x :int,y :int,weight :int,height :int):
    # (h , w) = diff.shape
    # print(h)
    # print(w)
    # for line in diff:
    #     for pix in line:
    #         if(pix != 255):
    #             print(pix)
    #             pix = 125

    #修改这个区域内的像素
    i = x
    j = y
    w = i + weight
    h = j + height
    for i in range(w):
        for j in range(h):
            if(diff[j][i] <= 250):
                # print(diff[j][i])
                diff[j][i]=100
            else:
                diff[j][i]=255

    # (h , w) = diff.shape
    # for i in range(w):
    #     for j in range(h):
    #         if(diff[j][i] <= 250):
    #             # print(diff[j][i])
    #             diff[j][i]=100
    #         else:
    #             diff[j][i]=255

def rotate_image(input_image, angle):
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
    # cv2.imwrite(f"huidus//test.png", output_image)
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
    # cv2.imwrite(f"huidus//test.png", res2)
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

# img = Image.open("huidus//7_frame.png")
# img = skimage.io.imread("huidus//7_frame.png")#读入图片
# out = np.array(img)
# out = out.astype(np.float64) / 255
# print(out.shape)
# rotate_image(out,45)
#np.pi/2
# cv2.imwrite(f"huidus//test.png", image)


def draw_rect(size,angle, save_path):
    thickness = 26

    imgbase = Image.open("frame_base.png")
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
    img = np.array(img_new)
    cv2.imwrite(save_path, rotate_image(img,angle))
    # img_new.save(save_path)
def draw_elliptic(size,angle, save_path):
    thickness = 26
    # print(size , angle ,save_path)
    new_size = (size[0]+26,size[1]+26)
    img_new = Image.new('RGBA', new_size)
    img = np.array(img_new)
    new_width, new_height = size
    x, y = new_size
    # shape = (size,4)
    # img = numpy.zeros(shape, dtype=float, order='C')
    cv2.ellipse(img, ((x/2,y/2),(new_width,new_height),0), (95, 235, 95, 255), thickness)  # 椭圆
    cv2.imwrite(save_path, rotate_image(img,angle))

def get_focus(center_pos,size,angle): #获得椭圆焦点
    short,long = size
    center_x,center_y = center_pos
    focus_len = math.pow(long*long - short*short,0.5)
    ang = 180 - angle
    # print(math.cos(math.radians(ang)),math.sin(math.radians(-ang)))
    radians1 = math.radians(-ang)
    radians2 = math.radians(ang)
    x0 = focus_len * math.sin(radians1)
    y0 = focus_len * math.cos(radians2) #以原点为中心点的时候
    # print("x0,y0",x0,y0,focus_len,angle)
    y1 = -y0 + center_y
    x1 = x0 + center_x
    y2 = y0 + center_y
    x2 = -x0 + center_x

    return [(x1,y1),(x2,y2)]
def is_in_elliptic(point,focus,long):
    focus1,focus2 = focus
    len1 = get_len(point,focus1)
    len2 = get_len(point,focus2)
    if len1+len2 <= long:
        return True
    return False
def is_in_circle(point,center,r):
    len = get_len(point,center)
    if len <= r:
        return True
    return False
def get_len(pos1,pos2):
    x1,y1 = pos1
    x2,y2 = pos2
    dx = x1 - x2
    dy = y1 - y2
    return math.pow(dx*dx + dy*dy,0.5)

psd = PSDImage.open("1.psd")
# layer1 = "1"
# i = 1
# for layer in psd:
    # print(layer.kind)
    # print(layer.mask)
    # print(layer.has_mask())
    # print(layer.offset)
    # print(layer)
    # if layer.is_group():
    #     continue
    # layer.name = f"test_{i}"
        # for child in layer:
        #     print(child)

print(psd.size)
# psd.save("out.psd")

# img_png = psd.composite()
# img_png.save('test_psd.png')

# layer1_png = psd[1].composite()  可以直接得到抠图
# layer1_png.save('test_psd.png')

# numpy_image = psd.numpy()
# layer_image = []
# for layer in psd:
#     if layer.is_group():
#         continue
#     layer_image.append(layer.numpy())
# cv2.imshow("test", numpy_image)
# cv2.waitKey(0)

img_num_base = None
img_base = None
layer_list = ['1','2','3','4','5','6','7','8','9','10']

for layer in psd:
    if layer.is_group():
        for baselayer in layer:
            if baselayer.name == "177":
                img_num_base = baselayer.numpy()
                img_base = baselayer.composite()
                # cv2.imshow("test1", img_num_base)
                # cv2.waitKey(0)
                # img_png = psd.composite()
                # img_png.save('test_psd.png')
    elif layer.name == "177":
        img_num_base = layer.numpy()
        img_base = layer.composite()
        break
if img_base is not None:
    img_num_base = np.array(img_base)  # imagebase转换成numpy格式
    gray_base = cv2.cvtColor(img_num_base, cv2.COLOR_RGBA2GRAY)
    img_png = psd.composite()
    img_num_all_mix = np.array(img_png)
    gray_all_mix = cv2.cvtColor(img_num_all_mix, cv2.COLOR_RGB2GRAY)
    (all_score, all_diff) = structural_similarity(gray_base, gray_all_mix, full=True)
    all_diff = (all_diff * 255).astype("uint8")
    all_thresh = cv2.threshold(all_diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_thresh1 = all_thresh.copy()  # 画椭圆的基础图

    base_h, base_w, baee_n = img_num_base.shape
    for curlayer in psd:
        if curlayer.is_group():
            continue
        elif curlayer.name in layer_list:
            img_mix = psd.composite(layer_filter= lambda layer: layer.name == curlayer.name or layer.name == '177') #or layer.name == '图层 0')
            img_num_mix = np.array(img_mix)
            # img_base.save('base_psd.png')
            # img_mix.save('mix_psd.png')
            gray_mix = cv2.cvtColor(img_num_mix, cv2.COLOR_RGB2GRAY)
            # cv2.imwrite("huidus//huidubase.png", gray_base)
            # cv2.imwrite("huidus//huidumix.png", gray_mix)
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

            minW = 300
            minH = 380
            minR = 160
            (r_x, r_y, r_w, r_h) = cv2.boundingRect(cnt)#普通外接矩形轮廓
            if r_w < r_h:
                if r_w < minW:
                    d_x = minW - r_w
                    r_x -= d_x/2
                    r_w = minW
                if r_h < minH:
                    d_y = minH - r_h
                    r_y -= d_y / 2
                    r_h = minH
            else:
                if r_h < minW:
                    d_y = minW - r_h
                    r_h = minW
                    r_y -= d_y / 2
                if r_w < minH:
                    d_x = minH - r_w
                    r_x -= d_x / 2
                    r_w = minH
            if r_x < 0: r_x = 0
            if r_y < 0: r_y = 0
            if r_x + r_w > base_w: r_x = base_w - r_w
            if r_y + r_h > base_h: r_y = base_h - r_h
            r_x = int(r_x)
            r_y = int(r_y)

            rect = cv2.minAreaRect(cnt)#最小矩形轮廓
            (center,(weight,height), angle) = rect
            old_weight,old_height = weight,height


            (e_x,e_y) , (e_a,e_b) , e_angle = cv2.fitEllipse(cnt) #椭圆轮廓 x, y）代表椭圆中心点的位置（a, b）代表长短轴长度，应注意a、b为长短轴的直径，而非半径 angle 代表了中心旋转的角度
            e_a = 1.15 * e_a #短轴
            e_b = 1.15 * e_b #长轴


            if weight < height:
                if weight < minW: weight = minW
                if height < minH: height = minH
            else:
                if height < minW: height = minW
                if weight < minH: weight = minH
            if e_a < minW: e_a = minW
            if e_b < minH: e_b = minH

            sR = weight * height
            rect = (center,(weight,height), angle)
            points = cv2.boxPoints(rect)
            points = np.int0(points)  # 最小矩形轮廓

            retval = (e_x,e_y) , (e_a,e_b) , e_angle #椭圆轮廓
            sE = np.pi * e_a/2 * e_b/2 #椭圆面积



            (c_x, c_y), radius = cv2.minEnclosingCircle(cnt) #圆轮廓
            center = (int(c_x), int(c_y))
            r = int(radius)
            oldr = r
            if r < minR:
                r = minR
            centerCut = (r, r)
            sC = r * r * math.pi


            out_path = f"huidus//mix_{curlayer.name}_frame.png"

            if sE < sR and sE < sC:
                is_draw_min_ellipse = True
                focus = get_focus((e_x, e_y), (e_a / 2, e_b / 2), e_angle)
                p1,p2 = np.int0(focus)
                print(curlayer.name,(e_x, e_y), (e_a, e_b), e_angle,focus)
                for i in range(base_w):
                    if not is_draw_min_ellipse:
                        print(curlayer.name , "椭圆超出边界")
                        break
                    pos0 = (i,0)
                    pos1 = (i,2000)
                    if is_in_elliptic(pos0,focus,e_b) or is_in_elliptic(pos1,focus,e_b):
                        is_draw_min_ellipse = False
                for i in range(base_h):
                    if not is_draw_min_ellipse:
                        print(curlayer.name , "椭圆超出边界")
                        break
                    pos0 = (0, i)
                    pos1 = (3000, i)
                    if is_in_elliptic(pos0,focus,e_b) or is_in_elliptic(pos1,focus,e_b):
                        is_draw_min_ellipse = False
                if is_draw_min_ellipse:
                    # cv2.line(img_num_all_mix,p1,p2,(95, 235, 95, 255), 10) #画出椭圆焦点连起来的线段
                    cv2.ellipse(all_thresh, retval, (95, 235, 95, 255), 26) #椭圆
                    cv2.ellipse(img_num_all_mix, retval, (95, 235, 95, 255), 26)  # 椭圆
                    draw_elliptic((int(e_a),int(e_b)),180 - int(e_angle), out_path) #切出椭圆
                    # print(curlayer.name,(e_x, e_y), (e_a, e_b), e_angle)
                else:
                    cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                    cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画外接矩形
                    size = (int(r_w), int(r_h))
                    draw_rect(size, 0, out_path)  # 切出矩形
            elif sC <= sE and sC <= sR:
                # print(curlayer.name,'画圆')
                is_draw_min_circle = True
                for i in range(base_w):
                    if not is_draw_min_circle:
                        print(curlayer.name, "圆超出边界")
                        break
                    pos0 = (i,0)
                    pos1 = (i,2000)
                    if is_in_circle(pos0,center,r) or is_in_circle(pos1,center,r):
                        is_draw_min_circle = False
                for i in range(base_h):
                    if not is_draw_min_circle:
                        print(curlayer.name, "圆超出边界")
                        break
                    pos0 = (0, i)
                    pos1 = (3000, i)
                    if is_in_circle(pos0, center, r) or is_in_circle(pos1,center,r):
                        is_draw_min_circle = False
                if is_draw_min_circle:
                    cv2.circle(all_thresh, center, r, (95, 235, 95), 26)  # 画外接圆
                    cv2.circle(img_num_all_mix, center, r, (95, 235, 95), 26)  # 画外接圆
                    imageCircle = np.zeros((2 * r, 2 * r, 4))  # 创建opencv图像
                    imageCircle[:] = (0, 0, 0, 0)
                    cv2.circle(imageCircle, centerCut, r - 21, (95, 235, 95, 255), 42)  # 画每个抠图的圆边框
                    cv2.imwrite(out_path, imageCircle)
                else:
                    cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                    cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画外接矩形
                    size = (int(r_w), int(r_h))
                    draw_rect(size, 0, out_path)  # 切出矩形
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
                    cv2.drawContours(all_thresh, [points], 0, (95, 235, 95, 255), 26) # 最小矩形
                    cv2.drawContours(img_num_all_mix, [points], 0, (95, 235, 95, 255), 26)  # 最小矩形
                    size = (int(weight), int(height))
                    # out_path = f"huidus//mix_{curlayer.name}_frame.png"
                    draw_rect(size, 180 - int(angle), out_path) #切出矩形
                else:
                    cv2.rectangle(all_thresh, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画普通外接矩形
                    cv2.rectangle(img_num_all_mix, (r_x, r_y), (r_x + r_w, r_y + r_h), (95, 235, 95), 10)  # 画外接矩形
                    size = (int(r_w), int(r_h))
                    draw_rect(size, 0, out_path) #切出矩形


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


            # cv2.imwrite("huidus//huidu111.png", thresh)
            # retval = cv2.moments(thresh.copy())
            # print(len(cnts))
    cv2.imwrite(f"huidus//huiduoooo.png", all_thresh)
    # cv2.imwrite(f"huidus//huidu1111.png", all_thresh1)
    cv2.imwrite(f"huidus//mixoooo.png", img_num_all_mix)
    img_png = psd.composite()
    # img_png.save('huidus//mix.png')
    # img_base.save('huidus//base.png')

    # print(retval)
    # cv2.drawContours(img_num_mix, retval, -1, (255, 255, 255), 5)
    # cv2.imshow("test1", img_num_mix)
    # cv2.waitKey(0)

    # hull = cv2.convexHull(points[, clockwise[, returnPoints]
    # rect = cv2.minAreaRect(diff)  # 得到面积最小的外接矩形

    # for i in range(0, len(cnts)):

# cnts, hierarchy = cv2.findContours(imageA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
