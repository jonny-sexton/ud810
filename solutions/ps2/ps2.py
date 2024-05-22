# ps2
import os
import numpy as np
import cv2 as cv

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

## 1-a
# Read images
L = cv.imread('problem-sets/ps2/input/pair0-L.png').astype("float32") * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv.imread('problem-sets/ps2/input/pair0-R.png').astype("float32") * (1.0 / 255.0)

D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)

# D_L = disparity_ncorr(L, R)
# D_R = disparity_ncorr(R, L)

# # display images
# # cv.imshow("", D_L)
# # cv.waitKey(0)
# # cv.imshow("", D_R)
# # cv.waitKey(0)

# # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# # Note: They may need to be scaled/shifted before saving to show results properly
cv.imwrite("problem-sets/ps2/output/ps2-1-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-1-a-2.png", D_R)

# TODO: Rest of your code here
L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE) * (1.0 / 255.0)

D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)

# display images
# cv.imshow("", D_L)
# cv.waitKey(0)
# cv.imshow("", D_R)
# cv.waitKey(0)

cv.imwrite("problem-sets/ps2/output/ps2-2-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-2-a-2.png", D_R)

## 2-b
"""
My result is a lot more noisy, e.g. at object edges. Window size and similarity function (in this case SSD) have an effect on the output, after research I have determined that a window size of [15,15] gives the best results. Furthermore, there is occlusion at object edges as certain regions are only visible in one image but not the other.
"""

## 3-a
L_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE), (9,9), 0)
R_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE), (9,9), 0)

# cv.imshow("", L_blur)
# cv.waitKey(0)
# cv.imshow("", R_blur)
# cv.waitKey(0)

D_L = disparity_ssd(L_blur * (1.0 / 255.0), R_blur * (1.0 / 255.0))
D_R = disparity_ssd(R_blur * (1.0 / 255.0), L_blur * (1.0 / 255.0))

cv.imwrite("problem-sets/ps2/output/ps2-3-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-3-a-2.png", D_R)

# Text response
"""
Gaussian noise was added to both images (A sigma of 9 was used). The output is not as smooth as the original and contains more noise.
"""

## 3-b
L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE)
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE)

L_cont = cv.convertScaleAbs(L, alpha=1.1, beta=0) # check if this increases contrast

# cv.imshow("", L)
# cv.waitKey(0)
# cv.imshow("", L_cont)
# cv.waitKey(0)

D_L = disparity_ssd(L_cont * (1.0 / 255.0), R * (1.0 / 255.0))
D_R = disparity_ssd(R * (1.0 / 255.0), L_cont * (1.0 / 255.0))

cv.imwrite("problem-sets/ps2/output/ps2-3-b-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-3-b-2.png", D_R)

# Text response
"""
Again, the output is not as smooth as the original and contains more noise.
"""

## 4-a
L = cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

# cv.imshow("", L)
# cv.waitKey(0)
# cv.imshow("", R)
# cv.waitKey(0)

D_L = disparity_ncorr(L, R)
D_R = disparity_ncorr(R, L)

print("4-a finished")

cv.imwrite("problem-sets/ps2/output/ps2-4-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-4-a-2.png", D_R)

# Text response
"""
Results are improved compared to SSD method. However there is still a lot of noise, and the matching algorithm does not handle occlusions well. Due to this, there are a lot of noisy regions in occluded areas of the image, and the dynamic range of the disparity map is affected by noisy maxima and minima, making it hard to distinguish between individual objects.
"""

## 4-b
L_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-L.png', cv.IMREAD_GRAYSCALE), (9,9), 0).astype("float32") * (1.0 / 255.0)
R_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE), (9,9), 0).astype("float32") * (1.0 / 255.0)

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
R = cv.imread('problem-sets/ps2/input/pair1-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

L_cont = cv.convertScaleAbs(L, alpha=1.1, beta=0).astype("float32") * (1.0 / 255.0) 

# cv.imshow("", L)
# cv.waitKey(0)
# cv.imshow("", L_cont)
# cv.waitKey(0)

D_L = disparity_ncorr(L_cont, R)
D_R = disparity_ncorr(R , L_cont)

print("4-b-2 finished")

cv.imwrite("problem-sets/ps2/output/ps2-4-b-3.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-4-b-4.png", D_R)

# Text response
"""
The images with a Gaussian are again noisier than the original. However, the contrasted set of images appear to be unaffected by the change in contrast. The disparity maps are very similar with only a negligible difference between the two. Therefore, the template matching algorithm can be considered robust to changes in contrast.
"""


## 5-a

L = cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
R = cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

D_L = disparity_ncorr(L, R)
D_R = disparity_ncorr(R, L)

print("5-a finished")

cv.imwrite("problem-sets/ps2/output/ps2-5-a-1.png", D_L)
cv.imwrite("problem-sets/ps2/output/ps2-5-a-2.png", D_R)

# Text response
"""
Tests were run to find an optimal window size. After running tests, an optimal window size of [15,15] was determined. If the window is too small, then the computed disparity map becomes noisy because there are not enough features in the window for a succesful match. Inversely, if the window size is too big then disparity maps become "washed out" and lose high frequency detail. This is due to a larger window being more likely to contain a depth discontinuity, leading to an inaccurate disparity being computed and the disparity map becoming "washed out".
It is also important to consider the baseline between images. We want the disparity map to show objects that are further away as darker, and objects that are closer as brighter. Therefore, I added a flag to each function to indicate if the images are flipped around or not (default is L --> R), via the "inv=1" option. When this flag is used, then the computed disparity map is negated before being normalized to [0,255]. 
ADDENDUM: I added a "disp_range" parameter to my disparity_ncorr.py function. This limits the width of strip_R to disp_range, and prevents anomalies affecting the dynamic range of the compute disparity map. I computed a seperate disparity map for a range of "disp_range" values, and as the value increases, objects which are closer to the camera become smoother. The optimum value appears to be 250 pixels, which seems to be the approximate difference between the foremost objects in the R and L image. This value would of course change if the baseline between the two images changed as well.

I therefore achieved optimum results with a disp_range value of 250 and a template size of 15,15. These values are used as the defaults in my function.

See "ncc_test_1_L.png", "ncc_test_1_R.png", "ncc_test_2_L.png" and "ncc_test_2_R.png" for my best results!
"""


## Testing

### Frame Size Test

# frame_sizes = [9, 11, 13 ,15, 17, 19 ,21, 23, 25, 27, 29, 31]

# L = cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
# R = cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

# for frame_size in frame_sizes:
#     print("computing SSD for frame_size:", frame_size)

#     D_L_SSD = disparity_ssd(L, R, frame_size=frame_size)
#     D_R_SSD = disparity_ssd(R, L, frame_size=frame_size)

#     output_str_SSD_L = "problem-sets/ps2/output/ssd_" + str(frame_size) + "_L.png"
#     output_str_SSD_R = "problem-sets/ps2/output/ssd_" + str(frame_size) + "_R.png"

#     cv.imwrite(output_str_SSD_L, D_L_SSD)
#     cv.imwrite(output_str_SSD_R, D_R_SSD)

#     print("computing NCC for frame_size:", frame_size)

#     D_L_NCC = disparity_ncorr(L, R, frame_size=frame_size)
#     D_R_NCC = disparity_ncorr(R, L, frame_size=frame_size)

#     output_str_NCC_L = "problem-sets/ps2/output/ncc_" + str(frame_size) + "_L.png"
#     output_str_NCC_R = "problem-sets/ps2/output/ncc_" + str(frame_size) + "_R.png"

#     cv.imwrite(output_str_NCC_L, D_L_NCC)
#     cv.imwrite(output_str_NCC_R, D_R_NCC)

### R --> L Test

# L = cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
# R = cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

# print("computing SSD.")

# D_L_SSD = disparity_ssd(L, R) # L --> R
# D_R_SSD = disparity_ssd(R, L, inv = 1) # R --> L

# output_str_SSD_L = "problem-sets/ps2/output/ssd_L-R.png"
# output_str_SSD_R = "problem-sets/ps2/output/ssd_R-L.png"

# cv.imwrite(output_str_SSD_L, D_L_SSD)
# cv.imwrite(output_str_SSD_R, D_R_SSD)

# print("computing NCC.")

# D_L_NCC = disparity_ncorr(L, R) # L --> R
# D_R_NCC = disparity_ncorr(R, L, inv=1) # R --> L

# output_str_NCC_L = "problem-sets/ps2/output/ncc_L-R.png"
# output_str_NCC_R = "problem-sets/ps2/output/ncc_R-L.png"

# cv.imwrite(output_str_NCC_L, D_L_NCC)
# cv.imwrite(output_str_NCC_R, D_R_NCC)

### Smoothness

# blurs = [3, 5, 7, 9, 11, 13 ,15, 17, 19 ,21]

# for blur in blurs:
    
#     print("computing NCC using blur sigma:", blur)

#     L_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE), (blur,blur), 0).astype("float32") * (1.0 / 255.0)
#     R_blur = cv.GaussianBlur(cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE), (blur,blur), 0).astype("float32") * (1.0 / 255.0)

#     D_L_NCC = disparity_ncorr(L_blur, R_blur) # L --> R
#     D_R_NCC = disparity_ncorr(R_blur, L_blur, inv=1) # R --> L

#     output_str_NCC_L = "problem-sets/ps2/output/ncc_blur_" + str(blur) + "L.png"
#     output_str_NCC_R = "problem-sets/ps2/output/ncc_blur_" + str(blur) + "R.png"

#     cv.imwrite(output_str_NCC_L, D_L_NCC)
#     cv.imwrite(output_str_NCC_R, D_R_NCC)

### Disparity range test

# disp_ranges = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
# disp_ranges = [50, 100, 150, 200, 250]
# disp_ranges = [300, 350, 400]

# for disp_range in disp_ranges:

#     print("computing NCC using disp_range:", disp_range)

#     L = cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
#     R = cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

#     D_L_NCC = disparity_ncorr(L, R, disp_range=disp_range) # L --> R
#     D_R_NCC = disparity_ncorr(R, L, inv=1, disp_range=disp_range) # R --> L

#     output_str_NCC_L = "problem-sets/ps2/output/ncc_disp_range_" + str(disp_range) + "L.png"
#     output_str_NCC_R = "problem-sets/ps2/output/ncc_disp_range_" + str(disp_range) + "R.png"

#     cv.imwrite(output_str_NCC_L, D_L_NCC)
#     cv.imwrite(output_str_NCC_R, D_R_NCC)

L = cv.imread('problem-sets/ps2/input/pair2-L.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)
R = cv.imread('problem-sets/ps2/input/pair2-R.png', cv.IMREAD_GRAYSCALE).astype("float32") * (1.0 / 255.0)

D_L_NCC = disparity_ncorr(L, R) # L --> R
D_R_NCC = disparity_ncorr(R, L, inv=1) # R --> L

output_str_NCC_L = "problem-sets/ps2/output/ncc_test_2_L.png"
output_str_NCC_R = "problem-sets/ps2/output/ncc_test_2_R.png"

cv.imwrite(output_str_NCC_L, D_L_NCC)
cv.imwrite(output_str_NCC_R, D_R_NCC)