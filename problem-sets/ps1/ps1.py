import cv2 as cv
import numpy as np
from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks
from hough_lines_draw import hough_lines_draw
from hough_circles_acc import hough_circles_acc
from find_circles import find_circles
#  ps1

# #  1-a
# img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input0.png")

# #  TODO: Compute edge image img_edges
# edges = cv.Canny(img,100,200)

# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-1-a-1.png", edges)

# #  2-a
# # H, theta, rho = hough_lines_acc(cv.blur(edges, (5,5)))
# H, theta, rho = hough_lines_acc(edges)
# # H, theta, rho = hough_lines_acc(edges, Theta=[-90,0.5,89], RhoResolution=0.5)

# # convert H to normalised uint8 image
# H_img = H * 255/np.max(H)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-2-a-1.png", H_img.astype("uint8"))

# #  2-b
# peaks = hough_peaks(H,6)
# for peak in peaks:
#     cv.circle(H_img, (peak[0], peak[1]), 10, 255, 1)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-2-b-1.png", H_img.astype("uint8"))

# #  2-c
# hough_lines_draw(img, 'problem-sets/ps1/ps1_matlab_template/output/ps1-2-c-1.png', peaks, rho, theta)

# #  2-d
# """
# I used the following parameters:
# Theta=[0, 1, 180]
# ThetaBins=181
# RhoResolution=1
# RhoBins=1025
# number of peaks: 6
# Theta bins is one for each degree between 0° and 180°, which is enough resolution to cover all angles and simple enough to implement with int() for example.
# Rho bins is calculated from the maximum possible rho value which is length + width of image. This is then doubled to account for negative values.
# Number of peaks is manually set to 6 for the 6 lines in the image, this can be of course changed to find more/less lines in the image.
# """

# #  3-a
# img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input0-noise.png")
# img_smooth = cv.GaussianBlur(img, (9,9), 0)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-3-a-1.png", img_smooth)

# #  3-b
# edges = cv.Canny(img,100,200)
# edges_smooth = cv.Canny(img_smooth,100,200)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-3-b-1.png", edges)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-3-b-2.png", edges_smooth)

# # 3-c
# H, theta, rho = hough_lines_acc(edges_smooth)

# # convert H to normalised uint8 image
# H_img = H * 255/np.max(H)

# peaks = hough_peaks(H,8)
# for peak in peaks:
#     cv.circle(H_img, (peak[0], peak[1]), 10, 255, 1)
# cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-3-c-1.png", H_img.astype("uint8"))
# hough_lines_draw(img, 'problem-sets/ps1/ps1_matlab_template/output/ps1-3-c-2.png', peaks, rho, theta)

# """
# I just had to increase the GaussianBlur() kernel size up a notch! And increase the number of peaks to 10
# """

#  4-a
img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input1.png", cv.IMREAD_GRAYSCALE)
img_smooth = cv.GaussianBlur(img, (11,11), 0)
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-4-a-1.png", img_smooth)

#  4-b
edges_smooth = cv.Canny(img_smooth,100,200)
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-4-b-1.png", edges_smooth)
H, theta, rho = hough_lines_acc(edges_smooth)

# convert H to normalised uint8 image
H_img = H * 255/np.max(H)

peaks = hough_peaks(H, 5)
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-4-c-1.png", H_img.astype("uint8"))
hough_lines_draw(img, 'problem-sets/ps1/ps1_matlab_template/output/ps1-4-c-2.png', peaks, rho, theta)

#  4-c
"""
This part was hard. Turns out my model was working only for vertical and horizontal lines, but none between those angles. For some reason I had to shift the theta iterations to between -90 and 90 degrees, and then subtract the output theta values from 180 degrees at the end, to achieve the same output as OpenCV's HoughLines() function. But in the end this fixed my problems and detected the lines perfectly :-)
"""

#  5-a
H = 0
H, a, b, c = hough_circles_acc(edges_smooth, 50)
peaks = hough_peaks(H, 50)

H=H.copy()

H_img = H * 255/np.max(H)

# for peak in peaks:

#     print(peak)
#     cv.circle(H_img.copy(), (peak[0], 180 - peak[1]), 10, 255, 1)

cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-5-a-1.png", img_smooth.astype("uint8"))
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-5-a-2.png", edges_smooth.astype("uint8"))
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-5-a-3.png", H_img.T.astype("uint8"))

find_circles(img, "problem-sets/ps1/ps1_matlab_template/output/ps1-5-b-1.png", peaks, a, b, c)

#  5-b
"""
This procedure was surprisingly more normal. Just had to rotate the full 360° and make sure I plot everything correctly (and correct for that sneaky 180 - peak[1] ;-).
As the two images were mapped to each other, I could take directly the (x,y) coordinates from the local maxima in (a,b) space.
"""

#  6-a 
img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input2.png", cv.IMREAD_GRAYSCALE)
img_smooth = cv.GaussianBlur(img, (11,11), 0)

edges_smooth = cv.Canny(img_smooth,100,200)
H, theta, rho = hough_lines_acc(edges_smooth)

# convert H to normalised uint8 image
H_img = H * 255/np.max(H)

peaks = hough_peaks(H, 20)
hough_lines_draw(img, 'problem-sets/ps1/ps1_matlab_template/output/ps1-6-a-1.png', peaks, rho, theta)

#  6-b
"""
Other objects also appear as lines, as they are more generally the boundary between objects.
"""

#  6-c
img = 0
img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input2.png", cv.IMREAD_GRAYSCALE)
img_smooth = cv.GaussianBlur(img, (15,15), 0)

edges_smooth = cv.Canny(img_smooth,0,130)
# edges_smooth = cv.GaussianBlur(edges_smooth, (15,15), 0)
H, theta, rho = hough_lines_acc(edges_smooth)

# convert H to normalised uint8 image
H_img = H * 255/np.max(H)




peaks = hough_peaks(H, 20)
hough_lines_draw(img.copy(), 'problem-sets/ps1/ps1_matlab_template/output/ps1-6-a-1.png', peaks, rho, theta)

# try with different params
img_smooth = cv.GaussianBlur(img, (21,21), 0)
edges_smooth = cv.Canny(img_smooth,100,130)
H, theta, rho = hough_lines_acc(edges_smooth)
peaks = hough_peaks(H, 5)

hough_lines_draw(img_smooth, 'problem-sets/ps1/ps1_matlab_template/output/ps1-6-c-1.png', peaks, rho, theta)

#  7-a
img_smooth = cv.GaussianBlur(img, (13,13), 0)
edges_smooth = cv.Canny(img_smooth,100,200)
H, a, b, c = hough_circles_acc(edges_smooth, 30)
peaks = hough_peaks(H, 50)
for peak in peaks:
    print(peak)
find_circles(img.copy(), "problem-sets/ps1/ps1_matlab_template/output/ps1-7-a-1.png", peaks, a, b, c)
cv.imwrite("problem-sets/ps1/ps1_matlab_template/output/ps1-7-a-2.png", edges_smooth.astype("uint8"))

#  7-b
"""
First off, I set the minimum radius to 5, so that fine grain stuff e.g. noise wouldn't be searched for. Then I modified the algorithm to accomodate different radius sizes, by adding a 3rd dimension to the accumulator array, and for each edge pixel voting once for each radius size.

Unfortunately it does not perform well on the noisy background, in particular font serifs contain a lot of circles. It would be easier to take pictures on non noisy surfaces to reduce this negative effect
"""

#  8-a
img = 0
img = cv.imread("problem-sets/ps1/ps1_matlab_template/input/ps1-input3.png", cv.IMREAD_GRAYSCALE)
img_smooth = cv.GaussianBlur(img, (15,15), 0)

edges_smooth = cv.Canny(img_smooth,0,130)
# edges_smooth = cv.GaussianBlur(edges_smooth, (15,15), 0)
H, theta, rho = hough_lines_acc(edges_smooth)

# convert H to normalised uint8 image
H_img = H * 255/np.max(H)

image_8_a = img.copy()

peaks = hough_peaks(H, 10)
hough_lines_draw(img, 'problem-sets/ps1/ps1_matlab_template/output/ps1-8-a-1.png', peaks, rho, theta)

H, a, b, c = hough_circles_acc(edges_smooth, 30)
peaks = hough_peaks(H, 50)
for peak in peaks:
    print(peak)
find_circles(img, "problem-sets/ps1/ps1_matlab_template/output/ps1-8-a-2.png", peaks, a, b, c)

#  8-b
"""
add another dimension to account for ellipsoid (or "squashed circle") shape.
"""