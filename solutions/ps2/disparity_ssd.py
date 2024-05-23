import numpy as np
import cv2 as cv

def disparity_ssd(L, R, frame_size=15, inv=0):
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

    # iterate along each pixel in L, taking into account window size
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ssd_best = [-1, -1]

            # get L patch
            patch_L = L[max(0,i-patch_mid[0]):min(L.shape[0],i+patch_mid[0]), max(0,j-patch_mid[1]):min(L.shape[1],j+patch_mid[1])]

            # find L in R
            for k in range(R.shape[1]):
                # get R patch
                patch_R = R[max(0,i-patch_mid[0]):min(R.shape[0],i+patch_mid[0]), max(0,k-patch_mid[1]):min(R.shape[1],k+patch_mid[1])]
                
                # print(i,j,k)
                # print("patch_L:", max(0,i-patch_mid[0]), ":", min(L.shape[0],i+patch_mid[0]), ",", max(0,j-patch_mid[1]), ":", min(L.shape[1],j+patch_mid[1]))
                # print("patch_R:", max(0,i-patch_mid[0]), ":", min(R.shape[0],i+patch_mid[0]), ",", max(0,k-patch_mid[1]), ":", min(R.shape[1],k+patch_mid[1]))

                # check if patches have same size and then compare patches
                if patch_L.shape == patch_R.shape:
                    ssd = np.sum((patch_L - patch_R)**2)
                    # check if ssd is smaller than previous best
                    if ssd_best[0] == -1:
                        ssd_best = [ssd, k - j]
                    elif ssd < ssd_best[0]:
                        ssd_best = [ssd, k - j]

            # print(i,j,k,ssd_best[0])
            # if ssd_best[1] != 0:
            #     print(ssd_best[1])
            disp[i, j] = ssd_best[1]

    # shift disp to +ve range only if disp contains negative ints
    # if np.min(disp) < 0:
    #     disp += np.abs(np.min(disp))

    # flip matrix around 0 if R --> L
    if not inv:
        print("inverting", np.min(disp), np.max(disp))
        disp = np.negative(disp)
        print("inverted", np.min(disp), np.max(disp))

    # scale to [0,1] using minmax normalization
    # disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
    cv.normalize(disp, disp, 0, 255, cv.NORM_MINMAX)

    # convert to uint8
    disp = disp.astype("uint8")

    return disp