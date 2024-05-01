# ps2
import os
import numpy as np
import cv2 as cv

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

# ## 1-a
# # Read images
# L = cv.imread('problem-sets/ps2/input/pair0-L.png') * (1.0 / 255.0)  # grayscale, [0, 1]
# R = cv.imread('problem-sets/ps2/input/pair0-R.png') * (1.0 / 255.0)

# D_L = disparity_ssd(L, R)
# D_R = disparity_ssd(R, L)

# # display images
# # cv.imshow("", D_L)
# # cv.waitKey(0)
# # cv.imshow("", D_R)
# # cv.waitKey(0)

# # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# # Note: They may need to be scaled/shifted before saving to show results properly
# cv.imwrite("problem-sets/ps2/output/ps2-1-a-1.png", D_L)
# cv.imwrite("problem-sets/ps2/output/ps2-1-a-2.png", D_R)

# # TODO: Rest of your code here
# L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
# R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE) * (1.0 / 255.0)

# D_L = disparity_ssd(L, R)
# D_R = disparity_ssd(R, L)

# # display images
# # cv.imshow("", D_L)
# # cv.waitKey(0)
# # cv.imshow("", D_R)
# # cv.waitKey(0)

# cv.imwrite("problem-sets/ps2/output/ps2-2-a-1.png", D_L)
# cv.imwrite("problem-sets/ps2/output/ps2-2-a-2.png", D_R)

# ## 2-b
# """
# My result is a lot more noisy. There is also a lot more noise at object edges. Window size and similarity function (in this case SSD) have an effect on the output, and further research is required to find the optimal params.
# """

# ## 3-a
# L_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE), (9,9), 0)
# R_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE), (9,9), 0)

# # cv.imshow("", L_blur)
# # cv.waitKey(0)
# # cv.imshow("", R_blur)
# # cv.waitKey(0)

# D_L = disparity_ssd(L_blur * (1.0 / 255.0), R_blur * (1.0 / 255.0))
# D_R = disparity_ssd(R_blur * (1.0 / 255.0), L_blur * (1.0 / 255.0))

# cv.imwrite("problem-sets/ps2/output/ps2-3-a-1.png", D_L)
# cv.imwrite("problem-sets/ps2/output/ps2-3-a-2.png", D_R)

# # Text response
# """
# """

# ## 3-b
# L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE)
# R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE)

# L_cont = cv.convertScaleAbs(L, alpha=1.1, beta=0) # check if this increases contrast

# # cv.imshow("", L)
# # cv.waitKey(0)
# # cv.imshow("", L_cont)
# # cv.waitKey(0)

# D_L = disparity_ssd(L_cont * (1.0 / 255.0), R * (1.0 / 255.0))
# D_R = disparity_ssd(R * (1.0 / 255.0), L_cont * (1.0 / 255.0))

# cv.imwrite("problem-sets/ps2/output/ps2-3-b-1.png", D_L)
# cv.imwrite("problem-sets/ps2/output/ps2-3-b-2.png", D_R)

## 4-a
L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE)
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE)

# cv.imshow("", L)
# cv.waitKey(0)
# cv.imshow("", R)
# cv.waitKey(0)

D_L = disparity_ncorr(L, R)
D_R = disparity_ncorr(R, L)

print("4-a finished")

cv.imwrite("problem-sets/ps2/output/ps2-4-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-4-a-2.png", D_R)

## 4-b
L_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE), (9,9), 0)
R_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE), (9,9), 0)

# cv.imshow("", L_blur)
# cv.waitKey(0)
# cv.imshow("", R_blur)
# cv.waitKey(0)

D_L = disparity_ncorr(L_blur, R_blur)
D_R = disparity_ncorr(R_blur, L_blur)

print("4-b-1 finished")

cv.imwrite("problem-sets/ps2/output/ps2-4-b-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-4-b-2.png", D_R)

L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE)
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE)

L_cont = cv.convertScaleAbs(L, alpha=1.1, beta=0) # check if this increases contrast

# cv.imshow("", L)
# cv.waitKey(0)
# cv.imshow("", L_cont)
# cv.waitKey(0)

D_L = disparity_ncorr(L_cont, R)
D_R = disparity_ncorr(R , L_cont)

print("4-b-2 finished")

cv.imwrite("problem-sets/ps2/output/ps2-4-b-3.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-4-b-4.png", D_R)

## 5-a

L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE)
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE)

D_L = disparity_ncorr(L, R)
D_R = disparity_ncorr(R, L)

print("5-a finished")

cv.imwrite("problem-sets/ps2/output/ps2-5-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-5-a-2.png", D_R)

