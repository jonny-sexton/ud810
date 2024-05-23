import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def ps0_1_a():
    pass

def ps0_2_a():
    # swap blue and red pixels of image 1
    image_1_b = image_1[:,:,0]
    image_1_r = image_1[:,:,2]
    image_1_swapped = image_1.copy()
    image_1_swapped[:,:,0] = image_1_r.copy()
    image_1_swapped[:,:,2] = image_1_b.copy()
    cv.imwrite("solutions/ps0/output/ps0-2-a-1.png",image_1_swapped)

def ps0_2_b():
    # create monochrome image by selecting green channel
    image_1_g = image_1[:,:,1]
    cv.imwrite("solutions/ps0/output/ps0-2-b-1.png",image_1_g)

def ps0_2_c():
    image_1_r = image_1[:,:,2]
    cv.imwrite("solutions/ps0/output/ps0-2-c-1.png",image_1_r)

def ps0_2_d():
    pass

def ps0_3_a():
    image_1_g = image_1[:,:,1]
    image_2_gray = cv.cvtColor(image_2, cv.COLOR_RGB2GRAY)
    image_1_g_w = int(image_1_g.shape[0]/2)
    image_1_g_h = int(image_1_g.shape[1]/2)
    image_2_gray_w = int(image_2_gray.shape[0]/2)
    image_2_gray_h = int(image_2_gray.shape[1]/2)
    image_2_gray[(image_2_gray_w - 50): (image_2_gray_w + 50), (image_2_gray_h -50): (image_2_gray_h +50)] = image_1_g[(image_1_g_w - 50): (image_1_g_w + 50), (image_1_g_h - 50): (image_1_g_h +50)]
    cv.imwrite("solutions/ps0/output/ps0-3-a-1.png",image_2_gray)

def ps0_4_a():
    image_1_g = image_1[:,:,1]
    image_1_g_max = np.max(image_1_g)
    image_1_g_min = np.min(image_1_g)
    image_1_g_mean = np.mean(image_1_g)
    image_1_g_std = np.std(image_1_g)
    print("img1_green max:", image_1_g_max)
    print("img1_green min:", image_1_g_min)
    print("img1_green mean:", image_1_g_mean)
    print("img1_green std dev:", image_1_g_std)

def ps0_4_b():
    image_1_g = image_1[:,:,1]
    image_1_g_mean = np.mean(image_1_g)
    image_1_g_std = np.std(image_1_g)
    image_1_g = image_1[:,:,1]
    image_1_g = image_1_g.astype("float64")
    image_1_g -= image_1_g_mean
    image_1_g /= image_1_g_std
    image_1_g *= 10
    image_1_g += image_1_g_mean
    cv.imwrite("solutions/ps0/output/ps0-4-b-1.png",image_1_g)

def ps0_4_c():
    image_1_g = image_1[:,:,1]
    translation_matrix = np.float32([ [1,0,-2], [0,1,0] ])   
    image_1_g_t = cv.warpAffine(image_1_g, translation_matrix, (image_1_g.shape[1], image_1_g.shape[0])) 
    cv.imwrite("solutions/ps0/output/ps0-4-c-1.png",image_1_g_t)

def ps0_4_d():
    image_1_g = image_1[:,:,1]
    translation_matrix = np.float32([ [1,0,-2], [0,1,0] ])   
    image_1_g_t = cv.warpAffine(image_1_g, translation_matrix, (image_1_g.shape[1], image_1_g.shape[0])) 
    image_1_g_shifted = image_1_g.astype("float64") - image_1_g_t.astype("float64")
    image_1_g_shifted = normalize_0255(image_1_g_shifted).astype("uint8")
    cv.imwrite("solutions/ps0/output/ps0-4-d-1.png",image_1_g_shifted)

def ps0_5_a():
    image_1_noise = image_1.copy()
    image_1_noise_g = image_1_noise[:,:,1]
    image_1_noise_g = gaussian_filter(image_1_noise_g, 3)
    image_1_noise[:,:,1] = image_1_noise_g.copy()
    cv.imwrite("solutions/ps0/output/ps0-5-a-1.png",image_1_noise)

def ps0_5_b():
    image_1_noise = image_1.copy()
    image_1_noise_b = image_1_noise[:,:,0]
    image_1_noise_b = gaussian_filter(image_1_noise_b, 3)
    image_1_noise[:,:,0] = image_1_noise_b.copy()
    cv.imwrite("solutions/ps0/output/ps0-5-b-1.png",image_1_noise)

def ps0_5_c():
    pass

ps0_list = OrderedDict([('1a', ps0_1_a), ('2a', ps0_2_a), ('2b', ps0_2_b), ('2c', ps0_2_c), ('2d', ps0_2_d), ('3a', ps0_3_a), ('4a', ps0_4_a), ('4b', ps0_4_b), ('4c', ps0_4_c), ('4d', ps0_4_d), ('5a', ps0_5_a), ('5b', ps0_5_b), ('5c', ps0_5_c)])

if __name__=="__main__":
    image_1 = cv.imread("solutions/ps0/output/ps0-1-a-1.png") #bgr
    image_2 = cv.imread("solutions/ps0/output/ps0-1-a-2.png")

    # main()
    if len(sys.argv) == 2:
        if sys.argv[1] in ps0_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps0_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,2a,2b,2c,2d,3a,4a,4b,4c,4d,5a,5b,5c} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps0_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps0_list[idx]()