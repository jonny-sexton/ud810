import cv2 as cv
import numpy as np
from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks
from hough_lines_draw import hough_lines_draw
from hough_circles_acc import hough_circles_acc
from find_circles import find_circles
from filter_lines import filter_lines
from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def ps1_1_a():
    img = cv.imread("solutions/ps1/input/ps1-input0.png")
    edges = cv.Canny(img,100,200)
    cv.imwrite("solutions/ps1/output/ps1-1-a-1.png", edges)

def ps1_2_a():
    edges = cv.imread("solutions/ps1/output/ps1-1-a-1.png", cv.IMREAD_GRAYSCALE)
    H, theta, rho = hough_lines_acc(edges)
    # convert H to normalised uint8 image
    H_img = normalize_0255(H)
    cv.imwrite("solutions/ps1/output/ps1-2-a-1.png", H_img.astype("uint8"))

def ps1_2_b():
    edges = cv.imread("solutions/ps1/output/ps1-1-a-1.png", cv.IMREAD_GRAYSCALE)
    H, theta, rho = hough_lines_acc(edges)
    H_img = normalize_0255(H)
    peaks = hough_peaks(H,6)
    for peak in peaks:
        cv.circle(H_img, (peak[0], 180 - peak[1]), 10, 255, 1)
    cv.imwrite("solutions/ps1/output/ps1-2-b-1.png", H_img.astype("uint8"))

def ps1_2_c():
    img = cv.imread("solutions/ps1/input/ps1-input0.png")
    edges = cv.imread("solutions/ps1/output/ps1-1-a-1.png", cv.IMREAD_GRAYSCALE)
    H, theta, rho = hough_lines_acc(edges)
    peaks = hough_peaks(H,6)
    hough_lines_draw(img, 'solutions/ps1/output/ps1-2-c-1.png', peaks, rho, theta)

def ps1_2_d():
    pass

def ps1_3_a():
    img = cv.imread("solutions/ps1/input/ps1-input0-noise.png")
    img_smooth = cv.GaussianBlur(img, (9,9), 0)
    cv.imwrite("solutions/ps1/output/ps1-3-a-1.png", img_smooth)

def ps1_3_b():
    img = cv.imread("solutions/ps1/input/ps1-input0-noise.png")
    img_smooth = cv.GaussianBlur(img, (9,9), 0)
    edges = cv.Canny(img,100,200)
    edges_smooth = cv.Canny(img_smooth,100,200)
    cv.imwrite("solutions/ps1/output/ps1-3-b-1.png", edges)
    cv.imwrite("solutions/ps1/output/ps1-3-b-2.png", edges_smooth)

def ps1_3_c():
    img = cv.imread("solutions/ps1/input/ps1-input0-noise.png")
    edges_smooth = cv.imread("solutions/ps1/output/ps1-3-b-2.png", cv.IMREAD_GRAYSCALE)
    H, theta, rho = hough_lines_acc(edges_smooth)
    # convert H to normalised uint8 image
    H_img = normalize_0255(H)
    peaks = hough_peaks(H,6)
    for peak in peaks:
        cv.circle(H_img, (peak[0], 180 - peak[1]), 10, 255, 1)
    cv.imwrite("solutions/ps1/output/ps1-3-c-1.png", H_img.astype("uint8"))
    hough_lines_draw(img, 'solutions/ps1/output/ps1-3-c-2.png', peaks, rho, theta)

def ps1_4_a():
    img = cv.imread("solutions/ps1/input/ps1-input1.png", cv.IMREAD_GRAYSCALE)
    img_smooth = cv.GaussianBlur(img, (11,11), 0)
    cv.imwrite("solutions/ps1/output/ps1-4-a-1.png", img_smooth)

def ps1_4_b():
    img_smooth = cv.imread("solutions/ps1/output/ps1-4-a-1.png")
    edges_smooth = cv.Canny(img_smooth,100,200)
    cv.imwrite("solutions/ps1/output/ps1-4-b-1.png", edges_smooth)
    

def ps1_4_c():
    img = cv.imread("solutions/ps1/input/ps1-input1.png", cv.IMREAD_GRAYSCALE)
    img = cv.merge([img, img, img])
    edges_smooth = cv.imread("solutions/ps1/output/ps1-4-b-1.png", cv.IMREAD_GRAYSCALE)
    H, theta, rho = hough_lines_acc(edges_smooth)
    # convert H to normalised uint8 image
    H_img = normalize_0255(H)
    peaks = hough_peaks(H, 5)
    peaks = filter_lines(peaks, theta, rho, 5, 50)
    for peak in peaks:
        cv.circle(H_img, (peak[0], 180 - peak[1]), 10, 255, 1)
    cv.imwrite("solutions/ps1/output/ps1-4-c-1.png", H_img.astype("uint8"))
    hough_lines_draw(img, 'solutions/ps1/output/ps1-4-c-2.png', peaks, rho, theta)

def ps1_5_a():
    img = cv.imread("solutions/ps1/input/ps1-input1.png", cv.IMREAD_GRAYSCALE)
    img = cv.merge([img, img, img])
    img_smooth = cv.imread("solutions/ps1/output/ps1-4-a-1.png")
    edges_smooth = cv.imread("solutions/ps1/output/ps1-4-b-1.png", cv.IMREAD_GRAYSCALE)
    H, a, b, c = hough_circles_acc(edges_smooth, 20, min_radius=20)
    peaks = hough_peaks(H, 10)
    H_img = normalize_0255(H)
    for peak in peaks:
        cv.circle(img, (180 - peak[1], peak[0]), 20, color=(0,0,255) ,thickness=2)
    cv.imwrite("solutions/ps1/output/ps1-5-a-1.png", img_smooth.astype("uint8"))
    cv.imwrite("solutions/ps1/output/ps1-5-a-2.png", edges_smooth.astype("uint8"))
    cv.imwrite("solutions/ps1/output/ps1-5-a-3.png", img)
    
def ps1_5_b():
    img = cv.imread("solutions/ps1/input/ps1-input1.png")
    edges_smooth = cv.imread("solutions/ps1/output/ps1-4-b-1.png", cv.IMREAD_GRAYSCALE)
    H, a, b, c = hough_circles_acc(edges_smooth, 30, min_radius=20)
    peaks = hough_peaks(H, 14)
    find_circles(img, "solutions/ps1/output/ps1-5-b-1.png", peaks, a, b, c)

def ps1_6_a():
    img = cv.imread("solutions/ps1/input/ps1-input2.png")
    img_smooth = cv.GaussianBlur(img, (7,7), 0)

    edges_smooth = cv.Canny(img_smooth,50,100)
    H, theta, rho = hough_lines_acc(edges_smooth)

    # convert H to normalised uint8 image
    H_img = normalize_0255(H)

    peaks = hough_peaks(H, 20)
    hough_lines_draw(img_smooth, 'solutions/ps1/output/ps1-6-a-1.png', peaks, rho, theta)

def ps1_6_b():
    pass

def ps1_6_c():
    img = cv.imread("solutions/ps1/input/ps1-input2.png")
    img_smooth = cv.GaussianBlur(img, (7,7), 0)

    edges_smooth = cv.Canny(img_smooth,50,100)
    H, theta, rho = hough_lines_acc(edges_smooth)
    peaks = hough_peaks(H, 10)

    # filter out lines that are not parallel
    peaks = filter_lines(peaks, theta, rho, 5, 50)

    hough_lines_draw(img_smooth, 'solutions/ps1/output/ps1-6-c-1.png', peaks, rho, theta)

def ps1_7_a():
    img = cv.imread("solutions/ps1/input/ps1-input2.png")
    img_orig = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_smooth = cv.GaussianBlur(img, (7,7), 0)
    img_smooth = cv.erode(img, np.ones((5,5),np.uint8), 1)
    img_smooth = cv.GaussianBlur(img_smooth, (7,7), 0)
    edges_smooth = cv.Canny(img_smooth,55,55)
    # cv.imshow("",edges_smooth)
    # cv.waitKey(0)
    H, a, b, c = hough_circles_acc(edges_smooth, 50, min_radius=20)
    # convert H to normalised uint8 image

    peaks = hough_peaks(H, 20)
    # for peak in peaks:
    #     print(peak)
    find_circles(img_orig, "solutions/ps1/output/ps1-7-a-1.png", peaks, a, b, c)
    cv.imwrite("solutions/ps1/output/ps1-7-a-2.png", edges_smooth.astype("uint8"))

def ps1_7_b():
    pass

def ps1_8_a():
    img = cv.imread("solutions/ps1/input/ps1-input3.png")

    # get edges
    img_smooth = cv.erode(img, np.ones((5,5),np.uint8), 1)
    img_smooth = cv.GaussianBlur(img_smooth, (7,7), 0)
    edges_smooth = cv.Canny(img_smooth,55,55)
    H, theta, rho = hough_lines_acc(edges_smooth)

    peaks = hough_peaks(H, 10)
    peaks = filter_lines(peaks, theta, rho, 5, 50)
    hough_lines_draw(img, 'solutions/ps1/output/ps1-8-a-1.png', peaks, rho, theta)

    H, a, b, c = hough_circles_acc(edges_smooth, 50, min_radius=25)
    peaks = hough_peaks(H, 20, nhood_size=20)
    find_circles(img, "solutions/ps1/output/ps1-8-a-1.png", peaks, a, b, c)

def ps1_8_b():
    pass

def ps1_8_c():
    img = cv.imread("solutions/ps1/input/ps1-input3.png")

    # perform perspective transform on image to unsquash circles
    pts1 = np.float32([[114,33],[545,18],[0,284],[683,279]])
    pts2 = np.float32([[0,0],[683,0],[0,512],[683,512]])
     
    M = cv.getPerspectiveTransform(pts1,pts2)
    img = cv.warpPerspective(img,M,(683,512))

    cv.imwrite("solutions/ps1/output/ps1-8-c-2.png", img)

    # get edges
    img_smooth = cv.erode(img, np.ones((5,5),np.uint8), 1)
    img_smooth = cv.GaussianBlur(img_smooth, (7,7), 0)
    edges_smooth = cv.Canny(img_smooth,55,55)
    H, theta, rho = hough_lines_acc(edges_smooth)

    peaks = hough_peaks(H, 10)
    peaks = filter_lines(peaks, theta, rho, 5, 50)
    hough_lines_draw(img, 'solutions/ps1/output/ps1-8-c-1.png', peaks, rho, theta)

    H, a, b, c = hough_circles_acc(edges_smooth, 50, min_radius=25)
    peaks = hough_peaks(H, 20, nhood_size=20)
    find_circles(img, "solutions/ps1/output/ps1-8-c-1.png", peaks, a, b, c)

ps1_list = OrderedDict([('1a', ps1_1_a), ('2a', ps1_2_a), ('2b', ps1_2_b), ('2c', ps1_2_c), ('2d', ps1_2_d), ('3a', ps1_3_a), ('3b', ps1_3_b), ('3c', ps1_3_c), ('4a', ps1_4_a), ('4b', ps1_4_b), ('4c', ps1_4_c), ('5a', ps1_5_a), ('5b', ps1_5_b), ('6a', ps1_6_a), ('6b', ps1_6_b), ('6c', ps1_6_c), ('7a', ps1_7_a), ('7b', ps1_7_b), ('8a', ps1_8_a), ('8b', ps1_8_b), ('8c', ps1_8_c)])

if __name__=="__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] in ps1_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps1_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,2a,2b,2c,2d,3a,4a,4b,4c,4d,5a,5b,5c} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps1_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps1_list[idx]()