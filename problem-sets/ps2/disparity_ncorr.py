import numpy as np
import cv2 as cv

def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    patch = [9,9]
    patch_mid = [(patch[0] // 2), (patch[1] // 2)] # only works for odd sized patches

    # init disp
    disp = np.zeros((L.shape[0], L.shape[1]))

    # iterate along each pixel in L, taking into account window size
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):

            # get L patch
            patch_L = L[max(0,i-patch_mid[0]):min(L.shape[0],i+patch_mid[0]), max(0,j-patch_mid[1]):min(L.shape[1],j+patch_mid[1])]

            # get R strip
            strip_R = R[max(0,i-patch_mid[0]):min(R.shape[0],i+patch_mid[0]), :]

            # compute ncc
            ncc = cv.matchTemplate(strip_R, patch_L, cv.TM_CCOEFF_NORMED)

            # cv.imshow("",ncc)
            # cv.waitKey(0)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(ncc)
            top_left = max_loc # because cv.TM_CCOEFF_NORMED
            midpoint = (top_left[0] + patch_mid[0], top_left[1] + patch_mid[1])

            # print(i,j)
            # print(patch_mid)
            # print(midpoint)

            disp[i, j] = midpoint[0] - j

    # shift disp to +ve range only if disp contains negative ints
    if np.min(disp) < 0:
        disp += np.abs(np.min(disp))

    # scale to [0,1] using minmax normalization
    disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))

    # scale to [0,255]
    disp *= 254

    # convert to uint8
    disp = disp.astype("uint8")

    return disp