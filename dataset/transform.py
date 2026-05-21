import cv2
import random
import numpy as np
from PIL import Image


def ToSketch(img):
    rand = random.random()
    if rand < 0.05:
        img = np.array(img)
        img_inv = 255 - img
        img_blur = cv2.GaussianBlur(img_inv, ksize=(27, 27), sigmaX=0, sigmaY=0)
        img_blend = cv2.divide(img, 255 - img_blur, scale=256)
        img_blend = Image.fromarray(img_blend)
        img_ret = img_blend
    elif rand < 0.5:
        img = np.array(img)
        start_x = random.randint(0, 126)
        start_y = random.randint(0, 252)
        end_x = random.randint(18, 144 - start_x)
        end_y = random.randint(36, 288 - start_y)
        img_roi = img[start_y:start_y+end_y, start_x:start_x+end_x]
        img_inv = 255 - img_roi
        img_blur = cv2.GaussianBlur(img_inv, ksize=(17, 17), sigmaX=0, sigmaY=0)
        img_blend = cv2.divide(img_roi, 255 - img_blur, scale=256)
        img[start_y:start_y+end_y, start_x:start_x+end_x] = img_blend
        img = Image.fromarray(img)
        img_ret = img
    else:
        img_ret = img
    return img_ret