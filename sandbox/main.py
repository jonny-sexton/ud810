import cv2 as cv
import numpy as np

def downsample(img):
    # TODO: img_d = ? (pick alternate rows, cols: 1, 3, 5, ...)
    img = cv.resize(img, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=0.5, 
                       fy=0.5, 
                       interpolation=cv.INTER_NEAREST)
    
    return img

def blur_downsample(img):
    # TODO: img_bd = ? (blur by 5x5 gaussian, then downsample)
    img = cv.GaussianBlur(img, (5,5), 0)
    img = cv.resize(img, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=0.5, 
                       fy=0.5, 
                       interpolation=cv.INTER_NEAREST)
    
    return img

img = cv.imread('sandbox/penny-farthing.png')

img_d = downsample(img)    # 1/2 size
img_d = downsample(img_d)  # 1/4 size
img_d = downsample(img_d)  # 1/8 size

img_bd = blur_downsample(img)     # 1/2 size
img_bd = blur_downsample(img_bd)  # 1/4 size
img_bd = blur_downsample(img_bd)  # 1/8 size

cv.imshow("", np.hstack((img_d, img_bd))); cv.waitKey(0)

#imshow(imresize(img_d, size(img)))   # view downsampled image in original size
#imshow(imresize(img_bd, size(img)))  # compare with blurred & downsampled
