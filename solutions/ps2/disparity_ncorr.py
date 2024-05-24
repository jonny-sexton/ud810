import numpy as np
import cv2 as cv
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def disparity_ncorr(L, R, frame_size=15, inv=0, disp_range=250):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    # define patch size
    patch = [frame_size,frame_size]
    patch_mid = [(patch[0] // 2), (patch[1] // 2)] # only works for odd sized patches

    # init disp
    disp = np.zeros((L.shape[0], L.shape[1]))

    # convert L,R to float32 for matchTemplate
    L = L.astype("float32")
    R = R.astype("float32")

    # iterate along each pixel in L, taking into account window size
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            # get L patch
            patch_L = L[max(0,i-patch_mid[0]):min(L.shape[0],i+patch_mid[0]), max(0,j-patch_mid[1]):min(L.shape[1],j+patch_mid[1])]

            # get R strip
            if disp_range>0:
                strip_R = R[max(0,i-patch_mid[0]):min(R.shape[0],i+patch_mid[0]),max(0,j-(disp_range//2)):min(R.shape[1],j+(disp_range//2))]
            else:
                strip_R = R[max(0,i-patch_mid[0]):min(R.shape[0],i+patch_mid[0]),:]

            # compute ncc
            ncc = cv.matchTemplate(strip_R, patch_L, cv.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(ncc)
            top_left = max_loc # because cv.TM_CCOEFF_NORMED
            midpoint = [top_left[0] + patch_mid[0], top_left[1] + patch_mid[1]]
            midpoint[0] += max(0,j-(disp_range//2)) # compensate for cropped strip_R

            disp[i, j] = j - midpoint[0] # compute difference relative to j

    # flip matrix around 0 if R --> L
    if inv:
        disp = np.negative(disp)

    # scale to [0,255] 
    disp = normalize_0255(disp)

    # convert to uint8
    disp = disp.astype("uint8")

    return disp