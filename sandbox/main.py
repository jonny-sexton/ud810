import cv2 as cv
import numpy as np

import sys
sys.path.append('./solutions/utils/')
from toolbox import *

img = normalize_01(cv.imread('sandbox/penny-farthing.png', cv.IMREAD_GRAYSCALE).astype("float32"))

intergral_image = normalize_0255(cv.integral(img)).astype("uint8")
intergral_image = intergral_image[1:,1:]

cv.imshow("", intergral_image); cv.waitKey(0)

#imshow(imresize(img_d, size(img)))   # view downsampled image in original size
#imshow(imresize(img_bd, size(img)))  # compare with blurred & downsampled
