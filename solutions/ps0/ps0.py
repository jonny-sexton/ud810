import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter

def main():
    image_1 = cv.imread("problem-sets/ps0/input/ps0-1-a-1.png") #bgr
    image_2 = cv.imread("problem-sets/ps0/input/ps0-1-a-2.png")

    # 2.a. swap blue and red pixels of image 1
    image_1_b = image_1[:,:,0]
    image_1_r = image_1[:,:,2]
    image_1_swapped = image_1.copy()
    image_1_swapped[:,:,0] = image_1_r.copy()
    image_1_swapped[:,:,2] = image_1_b.copy()
    cv.imwrite("problem-sets/ps0/output/ps0-2-a-1.png",image_1_swapped)

    # 2.b. create monochrome image by selecting green channel
    image_1_g = image_1[:,:,1]
    cv.imwrite("problem-sets/ps0/output/ps0-2-b-1.png",image_1_g)

    # 2.c. 
    image_1_r = image_1[:,:,2]
    cv.imwrite("problem-sets/ps0/output/ps0-2-c-1.png",image_1_r)

    # 3.a.
    image_2_gray = cv.cvtColor(image_2, cv.COLOR_RGB2GRAY)
    image_1_g_w = int(image_1_g.shape[0]/2)
    image_1_g_h = int(image_1_g.shape[1]/2)
    image_2_gray_w = int(image_2_gray.shape[0]/2)
    image_2_gray_h = int(image_2_gray.shape[1]/2)
    image_2_gray[(image_2_gray_w - 50): (image_2_gray_w + 50), (image_2_gray_h -50): (image_2_gray_h +50)] = image_1_g[(image_1_g_w - 50): (image_1_g_w + 50), (image_1_g_h - 50): (image_1_g_h +50)]
    cv.imwrite("problem-sets/ps0/output/ps0-3-a-1.png",image_2_gray)

    # 4.a. 
    image_1_g_max = np.max(image_1_g)
    image_1_g_min = np.min(image_1_g)
    image_1_g_mean = np.mean(image_1_g)
    image_1_g_std = np.std(image_1_g)
    print(image_1_g_max,image_1_g_min, image_1_g_mean, image_1_g_std)

    # 4.b.
    image_1_g_copy = image_1_g.copy()
    image_1_g = image_1_g.astype("float64")
    image_1_g -= image_1_g_mean
    image_1_g /= image_1_g_std
    image_1_g *= 10
    image_1_g += image_1_g_mean
    cv.imwrite("problem-sets/ps0/output/ps0-4-b-1.png",image_1_g)

    # 4.c.
    image_1_g = image_1_g_copy
    translation_matrix = np.float32([ [1,0,-2], [0,1,0] ])   
    image_1_g_t = cv.warpAffine(image_1_g, translation_matrix, (image_1_g.shape[1], image_1_g.shape[0])) 
    cv.imwrite("problem-sets/ps0/output/ps0-4-c-1.png",image_1_g_t)

    # 4.d. 
    image_1_g_shifted = image_1_g - image_1_g_t
    cv.imwrite("problem-sets/ps0/output/ps0-4-d-1.png",image_1_g_shifted)

    # 5.a.
    image_1_noise = image_1.copy()
    image_1_noise_g = image_1_noise[:,:,1]
    image_1_noise_g = gaussian_filter(image_1_noise_g, 3)
    image_1_noise[:,:,1] = image_1_noise_g.copy()
    cv.imwrite("problem-sets/ps0/output/ps0-5-a-1.png",image_1_noise)

    # 5.b.
    image_1_noise = image_1.copy()
    image_1_noise_b = image_1_noise[:,:,0]
    image_1_noise_b = gaussian_filter(image_1_noise_b, 3)
    print(image_1_noise.dtype, image_1_noise_b.dtype)
    image_1_noise[:,:,0] = image_1_noise_b.copy()
    cv.imwrite("problem-sets/ps0/output/ps0-5-b-1.png",image_1_noise)

    print(image_1_noise_g.mean(), image_1_noise_b.mean())
    image_1_noise_g_dr = 20*np.log(image_1_noise_g.max() / image_1_noise_g.min())
    image_1_noise_b_dr = 20*np.log(image_1_noise_b.max() / image_1_noise_b.min())
    print(image_1_noise_g_dr, image_1_noise_b_dr)
    cv.imshow("", image_1_noise_g)
    cv.waitKey(0)
    cv.imshow("", image_1_noise_b)
    cv.waitKey(0)
    

if __name__=="__main__":
    main()