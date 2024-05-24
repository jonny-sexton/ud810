import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel, generic_gradient_magnitude
import math
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def img_grad(img, gauss_kern=-1, gauss_sigma=1, axis=2):
    # apply filter
    if gauss_kern==-1:
        pass
    else:
        img = cv.GaussianBlur(img, (gauss_kern,gauss_kern), gauss_sigma)

    # compute x axis diff
    img_filtered_x = sobel(img, 1)

    # compute y axis diff
    img_filtered_y = sobel(img, 0)

    # compute xy axis diff
    img_filtered_xy = np.arctan2(img_filtered_y, img_filtered_x)

    if axis==0:
        return img_filtered_x

    elif axis==1:
        return img_filtered_y
    
    elif axis==2:
        return img_filtered_xy

    else:
        print("Axis param must be either 0, 1 or 2!")
        return None

def M(img_x, img_y):
    I_x_2 = np.sum(img_x**2)
    I_y_2 = np.sum(img_y**2)
    I_x_y = np.sum(img_x*img_y)

    return np.array([[I_x_2, I_x_y], [I_x_y, I_y_2]])

def C(M, alpha=0.04):
    return np.linalg.det(M) - alpha * (M.trace()**2)

def harris_corners(img, gauss_kern=-1, gauss_sigma=1,  win_size=3, sigma=1.0, alpha=0.04):
    # check if image is float64
    if img.dtype != "float64":
        print("img type must be float64!")
        return None
    
    # compute window mid coord
    win_mid = (win_size // 2 ) + 1

    # normalize images to [0,1]
    img = normalize_01(img)

    # compute I_x, I_y, R, gkern
    I_x = img_grad(img, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=0)
    I_y = img_grad(img, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=1)
    R = np.zeros(img.shape)
    gkern = gaussian_kernel(l=win_size, sig=sigma)

    # iterate each pixel with window size (u, z)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # get patch around pixel from deriv images and gkern. this was a pain in the arse
            patch_x = I_x[max(0,i-win_mid+1):min(img.shape[0],i+win_mid), max(0,j-win_mid+1):min(img.shape[1],j+win_mid)]
            patch_y = I_y[max(0,i-win_mid+1):min(img.shape[0],i+win_mid), max(0,j-win_mid+1):min(img.shape[1],j+win_mid)]
            patch_gkern = gkern[max(0,win_mid-i-1):min(win_size,img.shape[0]-i+win_mid-1),max(0,win_mid-j-1):min(win_size,img.shape[1]-j+win_mid-1)]
             
            # apply gaussian weighted average to patches
            patch_x_conv = patch_x * patch_gkern
            patch_y_conv = patch_y * patch_gkern

            # compute harris corner for patch
            R[i, j] = C(M(patch_x_conv, patch_y_conv), alpha=alpha)

    # normalize as image
    R_img = R.copy()
    R_img += np.abs(np.min(R_img)) # shift to positive values
    R_img = normalize_0255(R_img)
    R_img = R_img.astype("uint8")

    return R, R_img

def thresh(R, thresh=1):
    return (R > thresh) * R

def nms(R, win_size=3):
    R_new = np.zeros(R.shape)

    # compute half window length
    win_mid = (win_size // 2) + 1

    n = np.count_nonzero(R)
    
    for i in range(n):
        # get max value
        max_values = np.where(R == np.max(R))
        
        # get x y of max value
        x = max_values[0][0]
        y = max_values[1][0]
        max_value = R[x][y]

        # break if all points have been suppressed
        if max_value == 0:
            break

        R_new[x][y] = 255

        # supress values around it
        R[max(0,x-win_mid):min(R.shape[0],x+win_mid),max(0,y-win_mid):min(R.shape[1],y+win_mid)] = 0

    n_new = np.count_nonzero(R_new)
    print(n_new, "features remaining.")

    return R_new

def keypoint_angle(img, features, diameter=9):
    # normalize image
    img=normalize_01(img)

    # get grad images
    I_x = img_grad(img, gauss_kern=3, gauss_sigma=3, axis=0)
    I_y = img_grad(img, gauss_kern=3, gauss_sigma=3, axis=1)

    keypoints = []
    
    # iterate keypoints, compute angle
    for i in range(len(features[0])):
        x = int(features[0][i])
        y = int(features[1][i])
        angle = math.atan2(I_y[x][y], I_x[x][y])
        
        keypoints.append(cv.KeyPoint(y, x, diameter, np.rad2deg(angle), 0, 0, -1))
    
    return keypoints

def ransac_trans(matches, points_1, points_2, error=3):
    k = 5000
    rng = np.random.default_rng()
    consensus_set = []
    consensus_set_id = -1

    for i in range(k):
        # pick random match
        id_rand = int(rng.random() * len(matches))

        # get point
        p1 = points_1[matches[id_rand].queryIdx].pt
        p2 = points_2[matches[id_rand].trainIdx].pt

        # get translation
        trans_x = int(p2[0]) - int(p1[0])
        trans_y = int(p2[1]) - int(p1[1])

        S_temp = np.array([[1, 0, trans_x], [0, 1, trans_y]])

        consensus_set_temp = []

        # check translation to all other matches
        for j in range(len(matches)):
            # do nothing for same match
            if i == j:
                continue
            
            # get temp point
            p1_temp = points_1[matches[j].queryIdx].pt
            p2_temp = points_2[matches[j].trainIdx].pt

            trans_x_temp = int(p2_temp[0]) - int(p1_temp[0])
            trans_y_temp = int(p2_temp[1]) - int(p1_temp[1])

            # check if transformation matches within error boundaries
            if (trans_x_temp > (trans_x - error)) and (trans_x_temp < (trans_x + error)):
                if (trans_y_temp > (trans_y - error)) and (trans_y_temp < (trans_y + error)):
                    consensus_set_temp.append(j)

        if len(consensus_set_temp) > len(consensus_set):
            S = S_temp
            consensus_set = consensus_set_temp

    return consensus_set, S

def ransac_sim(matches, points_1, points_2, error=3):
    k = 20000
    rng = np.random.default_rng()
    consensus_set = []

    for i in range(k):
        # pick two random matches
        id_rand_1 = int(rng.random() * len(matches))
        id_rand_2 = int(rng.random() * len(matches))

        # get points
        p1 = points_1[matches[id_rand_1].queryIdx].pt
        p2 = points_2[matches[id_rand_1].trainIdx].pt
        p3 = points_1[matches[id_rand_2].queryIdx].pt
        p4 = points_2[matches[id_rand_2].trainIdx].pt

        # find least squares solution for M
        u_1 = p1[0]
        v_1 = p1[1]
        u_2 = p2[0]
        v_2 = p2[1]
        u_3 = p3[0]
        v_3 = p3[1]
        u_4 = p4[0]
        v_4 = p4[1]

        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        
        A[0] = [u_1, -v_1, 1, 0]
        A[1] = [v_1, u_1, 0, 1]
        A[2] = [u_3, -v_3, 1, 0]
        A[3] = [v_3, u_3, 0, 1]

        B[0] = u_2
        B[1] = v_2
        B[2] = u_4
        B[3] = v_4

        # compute least squares
        M, res, _, _ = np.linalg.lstsq(A,B, rcond=None)

        a = M[0][0]
        b = M[1][0]
        c = M[2][0]
        d = M[3][0]

        S_temp = np.array([[a, -b, c], [b, a, d]])

        consensus_set_temp = []

        # compare transformation to all other matches
        for j in range(len(matches)):
            # get temp point
            p1_temp = np.append(np.array(points_1[matches[j].queryIdx].pt), 1)
            p2_temp = np.array(points_2[matches[j].trainIdx].pt)

            # transform point using M
            p2_temp_trans = S_temp @ p1_temp

            # check if transformation matches within error boundaries
            if (p2_temp_trans[0] > (p2_temp[0] - error)) and (p2_temp_trans[0] < (p2_temp[0] + error)):
                if (p2_temp_trans[1] > (p2_temp[1] - error)) and (p2_temp_trans[1] < (p2_temp[1] + error)):
                    consensus_set_temp.append(j)

        if len(consensus_set_temp) > len(consensus_set):
            consensus_set = consensus_set_temp
            S = S_temp

    return consensus_set, S

def ransac_aff(matches, points_1, points_2, error=3):
    k = 40000
    rng = np.random.default_rng()
    consensus_set = []

    for i in range(k):
        # pick three random matches
        id_rand_1 = int(rng.random() * len(matches))
        id_rand_2 = int(rng.random() * len(matches))
        id_rand_3 = int(rng.random() * len(matches))

        # get points
        p1 = points_1[matches[id_rand_1].queryIdx].pt
        p2 = points_2[matches[id_rand_1].trainIdx].pt
        p3 = points_1[matches[id_rand_2].queryIdx].pt
        p4 = points_2[matches[id_rand_2].trainIdx].pt
        p5 = points_1[matches[id_rand_3].queryIdx].pt
        p6 = points_2[matches[id_rand_3].trainIdx].pt

        # find least squares solution for M
        u_1 = p1[0]
        v_1 = p1[1]
        u_2 = p2[0]
        v_2 = p2[1]
        u_3 = p3[0]
        v_3 = p3[1]
        u_4 = p4[0]
        v_4 = p4[1]
        u_5 = p5[0]
        v_5 = p5[1]
        u_6 = p6[0]
        v_6 = p6[1]

        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        
        A[0] = [u_1, v_1, 1, 0, 0, 0]
        A[1] = [0, 0, 0, u_1, v_1, 1]
        A[2] = [u_3, v_3, 1, 0, 0, 0]
        A[3] = [0, 0, 0, u_3, v_3, 1]
        A[4] = [u_5, v_5, 1, 0, 0, 0]
        A[5] = [0, 0, 0, u_5, v_5, 1]

        B[0] = u_2
        B[1] = v_2
        B[2] = u_4
        B[3] = v_4
        B[4] = u_6
        B[5] = v_6

        # compute least squares
        M, res, _, _ = np.linalg.lstsq(A,B, rcond=None)

        a = M[0][0]
        b = M[1][0]
        c = M[2][0]
        d = M[3][0]
        e = M[4][0]
        f = M[5][0]

        S_temp = np.array([[a, b, c], [d, e, f]])

        consensus_set_temp = []

        # compare transformation to all other matches
        for j in range(len(matches)):
            # get temp point
            p1_temp = np.append(np.array(points_1[matches[j].queryIdx].pt), 1)
            p2_temp = np.array(points_2[matches[j].trainIdx].pt)

            # transform point using S
            p2_temp_trans = S_temp @ p1_temp

            # check if transformation matches within error boundaries
            if (p2_temp_trans[0] > (p2_temp[0] - error)) and (p2_temp_trans[0] < (p2_temp[0] + error)):
                if (p2_temp_trans[1] > (p2_temp[1] - error)) and (p2_temp_trans[1] < (p2_temp[1] + error)):
                    consensus_set_temp.append(j)

        if len(consensus_set_temp) > len(consensus_set):
            consensus_set = consensus_set_temp
            S = S_temp

    return consensus_set, S
