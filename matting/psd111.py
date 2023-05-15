import PIL.Image as Image
import numpy
from psd_tools import PSDImage
import imutils
import cv2
import os
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

def fill_all_pixels(diff: numpy.ndarray):
    (h , w) = diff.shape
    print(h)
    print(w)
    # for line in diff:
    #     for pix in line:
    #         if(pix != 255):
    #             print(pix)
    #             pix = 125
    for i in range(w):
        for j in range(h):
            if(diff[j][i] <= 240):
                # print(diff[j][i])
                diff[j][i]=100
            else:
                diff[j][i]=255





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
for layer in psd:
    if layer.is_group():
        for baselayer in layer:
            if baselayer.name == "177":
                print(11111)
                img_num_base = baselayer.numpy()
                img_base = baselayer.composite()
                # cv2.imshow("test1", img_num_base)
                # cv2.waitKey(0)
                # img_png = psd.composite()
                # img_png.save('test_psd.png')
    if layer.name == "177":
        print(2222)
        img_num_base = layer.numpy()
        img_base = layer.composite()
if img_base is not None:
    # img_num_mix = psd.numpy()
    img_mix = psd.composite(layer_filter= lambda layer: layer.name == "1"or layer.name == '177' ) #or layer.name == '图层 0')
    img_num_base = np.array(img_base)#image转换成numpy格式
    img_num_mix = np.array(img_mix)
    # img_base.save('base_psd.png')
    # img_mix.save('mix_psd.png')

    # cv2.imshow("test1", img_num_mix)
    # cv2.imshow("test2", img_num_base)
    # cv2.waitKey(0)
    gray_base = cv2.cvtColor(img_num_base, cv2.COLOR_RGBA2GRAY)
    gray_mix = cv2.cvtColor(img_num_mix, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("huidubase.png", gray_base)
    # cv2.imwrite("huidumix.png", gray_mix)
    (score, diff) = structural_similarity(gray_base, gray_mix, full=True)
    diff = (diff * 255).astype("uint8")
    # print(type(diff.shape))
    # print(type(diff.shape[0]))
    h = diff.shape[0]
    w = diff.shape[0]
    fill_all_pixels(diff)
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.imwrite("huidu111.png", thresh)
    retval = cv2.moments(thresh.copy())
    print(len(cnts))
    # print(retval)
    # cv2.drawContours(img_num_mix, retval, -1, (255, 255, 255), 5)
    # cv2.imshow("test1", img_num_mix)
    # cv2.waitKey(0)

    # hull = cv2.convexHull(points[, clockwise[, returnPoints]
    # rect = cv2.minAreaRect(diff)  # 得到面积最小的外接矩形

    # for i in range(0, len(cnts)):

# cnts, hierarchy = cv2.findContours(imageA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
