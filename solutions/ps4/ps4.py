import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel, generic_gradient_magnitude
from utils.toolbox import *
import math

def grad(img, gauss_kern=-1, gauss_sigma=1, axis=2):
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
        # img_filtered_x += np.abs(np.min(img_filtered_x)) # shift to positive values
        # img_filtered_x *= 255.0 / np.max(img_filtered_x) # normalize to [0, 255]
        # img_filtered_x = img_filtered_x.astype("uint8")
        return img_filtered_x

    elif axis==1:
        # img_filtered_y += np.abs(np.min(img_filtered_y))
        # img_filtered_y *= 255.0 / np.max(img_filtered_y)
        # img_filtered_y = img_filtered_y.astype("uint8")
        return img_filtered_y
    
    elif axis==2:
        # img_filtered_xy += np.abs(np.min(img_filtered_xy)) # shift values
        # img_filtered_xy *= 255.0 / np.max(img_filtered_xy) # normalize [0,255]
        # img_filtered_xy = img_filtered_xy.astype("uint8")
        return img_filtered_xy

    else:
        print("Axis param must be either 0, 1 or 2!")
        return None

def M(img_x, img_y):
    I_x_2 = np.sum(img_x**2)
    I_y_2 = np.sum(img_y**2)
    I_x_y = np.sum(img_x*img_y)

    # return np.array([[I_x_2, 0], [0, I_y_2]])
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
    I_x = grad(img, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=0)
    I_y = grad(img, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=1)
    R = np.zeros(img.shape)
    gkern = gaussian_kernel(l=win_size, sig=sigma)

    # cv.imshow('', np.hstack((normalize_0255(I_x).astype("uint8"),normalize_0255(I_y).astype("uint8")))); cv.waitKey(0); cv.destroyAllWindows()

    # iterate each pixel with window size (u, z)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # print("j:",j)
            # get patch around pixel from deriv images and gkern. this was a pain in the arse
            patch_x = I_x[max(0,i-win_mid+1):min(img.shape[0],i+win_mid), max(0,j-win_mid+1):min(img.shape[1],j+win_mid)]
            patch_y = I_y[max(0,i-win_mid+1):min(img.shape[0],i+win_mid), max(0,j-win_mid+1):min(img.shape[1],j+win_mid)]
            patch_gkern = gkern[max(0,win_mid-i-1):min(win_size,img.shape[0]-i+win_mid-1),max(0,win_mid-j-1):min(win_size,img.shape[1]-j+win_mid-1)]

            # cv.imshow('', np.hstack((normalize_0255(patch_x).astype("uint8"),normalize_0255(patch_y).astype("uint8"),normalize_0255(patch_gkern).astype("uint8")))); cv.waitKey(0); cv.destroyAllWindows()
             
            # apply gaussian weighted average to patches
            patch_x_conv = patch_x * patch_gkern
            patch_y_conv = patch_y * patch_gkern

            # cv.imshow('', np.hstack((normalize_0255(patch_x_conv).astype("uint8"), normalize_0255(patch_y_conv).astype("uint8")))); cv.waitKey(0); cv.destroyAllWindows()

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

    # get gradient images
    I_x = grad(img, gauss_kern=3, gauss_sigma=3, axis=0)
    I_y = grad(img, gauss_kern=3, gauss_sigma=3, axis=1)

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
            consensus_set = consensus_set_temp
            consensus_set_id = id_rand

    return consensus_set_id, consensus_set

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

if __name__ == "__main__":
    ## 1-a
    # img = cv.imread("problem-sets/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    # img_x = grad(img, gauss=1, axis=0)
    # img_y = grad(img, gauss=1, axis=1)
    # img_xy = np.hstack((img_x,img_y))

    # cv.imshow('', img_xy); cv.waitKey(0); cv.destroyAllWindows()
    # cv.imwrite("problem-sets/ps4/output/ps4-1-a-1.png", img_xy)

    # img = cv.imread("problem-sets/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    # img_x = grad(img, gauss=1, axis=0)
    # img_y = grad(img, gauss=1, axis=1)
    # img_xy = np.hstack((img_x,img_y))

    # cv.imshow('', img_xy); cv.waitKey(0); cv.destroyAllWindows()
    # cv.imwrite("problem-sets/ps4/output/ps4-1-a-2.png", img_xy)

    ## 1-b
    img_1 = cv.imread("problem-sets/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_1 = harris_corners(img_1.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    R_1_diff = R_1.copy()
    # cv.imwrite("problem-sets/ps4/output/ps4-1-b-1.png", R_1)

    img_2 = cv.imread("problem-sets/ps4/input/transB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_2 = harris_corners(img_2.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-b-2.png", R_2)

    img_3 = cv.imread("problem-sets/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_3 = harris_corners(img_3.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-b-3.png", R_3)

    img_4 = cv.imread("problem-sets/ps4/input/simB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_4 = harris_corners(img_4.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-b-4.png", R_4)

    # 1-c
    # img = cv.imread("problem-sets/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    # R, R_img = harris_corners(img.astype("float64").copy())

    print("thresholding...")
    # treshold
    R_1 = thresh(R_1, thresh=math.ceil(np.average(R_1)))
    R_2 = thresh(R_2, thresh=math.ceil(np.average(R_2)))
    R_3 = thresh(R_3, thresh=math.ceil(np.average(R_3)))
    R_4 = thresh(R_4, thresh=math.ceil(np.average(R_4)))

    print("NMS...")
    # NMS
    R_1 = nms(R_1, win_size=9)
    R_2 = nms(R_2, win_size=9)
    R_3 = nms(R_3, win_size=9)
    R_4 = nms(R_4, win_size=9)

    print("drawing corners...")
    # draw corners
    # img_1 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR) # convert to bgr
    # img_2 = cv.cvtColor(img_2, cv.COLOR_GRAY2BGR) # convert to bgr
    # img_3 = cv.cvtColor(img_3, cv.COLOR_GRAY2BGR) # convert to bgr
    # img_4 = cv.cvtColor(img_4, cv.COLOR_GRAY2BGR) # convert to bgr

    corners_1 = np.nonzero(R_1)
    corners_2 = np.nonzero(R_2)
    corners_3 = np.nonzero(R_3)
    corners_4 = np.nonzero(R_4)

    # for i in range(len(corners_1[0])):
    #     cv.circle(img_1, (corners_1[1][i], corners_1[0][i]), 0, (0, 0, 255), 1)
    # for i in range(len(corners_2[0])):
    #     cv.circle(img_2, (corners_2[1][i], corners_2[0][i]), 0, (0, 0, 255), 1)
    # for i in range(len(corners_3[0])):
    #     cv.circle(img_3, (corners_3[1][i], corners_3[0][i]), 0, (0, 0, 255), 1)
    # for i in range(len(corners_4[0])):
    #     cv.circle(img_4, (corners_4[1][i], corners_4[0][i]), 0, (0, 0, 255), 1)

    # cv.imshow('', np.hstack((img_1,img_2,img_3,img_4))); cv.waitKey(0); cv.destroyAllWindows()
    # R_1 = cv.cvtColor(R_1.astype("uint8"),cv.COLOR_GRAY2BGR)
    # cv.imshow('', np.hstack((cv.cvtColor(R_1_diff, cv.COLOR_GRAY2BGR), R_1, img_1))); cv.waitKey(0); cv.destroyAllWindows()

    # cv.imwrite("problem-sets/ps4/output/ps4-1-c-1.png", img_1)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-c-2.png", img_2)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-c-3.png", img_3)
    # cv.imwrite("problem-sets/ps4/output/ps4-1-c-4.png", img_4)

    """
    TEXT RESPONSE:
    - Setting the Sigma to 2 for computing the Harris value (sigma=2) made a huge difference. Then just need to fine tune the threshold to get as many features as possible.
    - Finding the maximum number of features from the Harris value map was improved massively by using the mean pixel intensity to compute the threshold value.
    - The rotated image does not behave so well. I imagine this is because the partial derivatives I_x, I_y become I_xy. This value is set to zero in the M matrix. Furthermore, due to sampling issues, the diagonal lines create a lot of small derivatives in the I_x, I_y direction.
    """

    ## 2-a
    print("computing keypoint angles...")
    keypoints_1 = keypoint_angle(img_1, corners_1)
    keypoints_2 = keypoint_angle(img_2, corners_2)
    keypoints_3 = keypoint_angle(img_3, corners_3)
    keypoints_4 = keypoint_angle(img_4, corners_4)
    
    # # draw keypoints
    # img_1 = cv.drawKeypoints(img_1, keypoints_1, img_1, flags=4)
    # img_2 = cv.drawKeypoints(img_2, keypoints_2, img_2, flags=4)
    # img_3 = cv.drawKeypoints(img_3, keypoints_3, img_3, flags=4)
    # img_4 = cv.drawKeypoints(img_4, keypoints_4, img_4, flags=4)
    
    # cv.imshow('', np.hstack((img_1,img_2))); cv.waitKey(0); cv.destroyAllWindows()
    # cv.imshow('', np.hstack((img_3,img_4))); cv.waitKey(0); cv.destroyAllWindows()

    # cv.imwrite("problem-sets/ps4/output/ps4-2-a-1.png", np.hstack((img_1,img_2)))
    # cv.imwrite("problem-sets/ps4/output/ps4-2-a-2.png", np.hstack((img_3,img_4)))

    ## 2-b
    # get descriptors
    print("computing descriptors...")
    sift = cv.SIFT_create()
    # keypoints_1 = tuple(keypoints_1)
    # keypoints_2 = tuple(keypoints_2)
    points_1, descriptors_1 = sift.compute(img_1,keypoints_1)
    points_2, descriptors_2 = sift.compute(img_2,keypoints_2)
    points_3, descriptors_3 = sift.compute(img_3,keypoints_3)
    points_4, descriptors_4 = sift.compute(img_4,keypoints_4)

    # compute matches
    print("computing matches...")
    bfm = cv.BFMatcher()
    matches_1 = bfm.match(descriptors_1, descriptors_2)
    matches_2 = bfm.match(descriptors_3, descriptors_4)
    
    # join images
    img_pair_1 = cv.cvtColor(np.hstack((img_1,img_2)),cv.COLOR_GRAY2BGR)
    img_pair_2 = cv.cvtColor(np.hstack((img_3,img_4)),cv.COLOR_GRAY2BGR)

    img_pair_matches_1 = img_pair_1.copy()
    img_pair_matches_2 = img_pair_2.copy()

    # draw matches
    for match in matches_1:
        p1 = points_1[match.queryIdx].pt
        p2 = points_2[match.trainIdx].pt
        cv.line(img_pair_matches_1, (int(p1[0]),int(p1[1])), (int(p2[0]+img_2.shape[1]),int(p2[1])), (0,0,255),1)

    for match in matches_2:
        p1 = points_3[match.queryIdx].pt
        p2 = points_4[match.trainIdx].pt
        cv.line(img_pair_matches_2, (int(p1[0]),int(p1[1])), (int(p2[0]+img_4.shape[1]),int(p2[1])), (0,0,255),1)

    # cv.imshow("",img_pair_matches_1)
    # cv.waitKey(0)

    # cv.imshow("",img_pair_matches_2)
    # cv.waitKey(0)

    # cv.imwrite("problem-sets/ps4/output/ps4-2-b-1.png", img_pair_matches_1)
    # cv.imwrite("problem-sets/ps4/output/ps4-2-b-2.png", img_pair_matches_2)

    ## 3-a 
    # compute ransac best transform
    # consensus_set_id, consensus_set = ransac_trans(matches_1, points_1, points_2)

    # for id in consensus_set:
    #     p1 = points_1[matches_1[id].queryIdx].pt
    #     p2 = points_2[matches_1[id].trainIdx].pt
    #     cv.line(img_pair_1, (int(p1[0]),int(p1[1])), (int(p2[0]+img_2.shape[1]),int(p2[1])), (0,0,255),1)

    # cv.imshow("",img_pair_1)
    # cv.waitKey(0)

    # cv.imwrite("problem-sets/ps4/output/ps4-3-a-1.png", img_pair_1)

    ## 3-b
    # compute ransac similarity transform
    print("computing ransac similarity transform")
    consensus_set_sim, S_sim = ransac_sim(matches_2, points_3, points_4, error=3)
    img_pair_2_sim = img_pair_2.copy()

    for id in consensus_set_sim:
        p1 = points_3[matches_2[id].queryIdx].pt
        p2 = points_4[matches_2[id].trainIdx].pt
        cv.line(img_pair_2_sim, (int(p1[0]),int(p1[1])), (int(p2[0]+img_4.shape[1]),int(p2[1])), (0,0,255),1)

    cv.imshow("",img_pair_2_sim)
    cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-b-1.png", img_pair_2_sim)

    ## 3-c
    # compute ransac affine transform
    print("computing ransac affine transform")
    consensus_set_aff, S_aff = ransac_aff(matches_2, points_3, points_4, error=3)
    img_pair_2_aff = img_pair_2.copy()

    for id in consensus_set_aff:
        p1 = points_3[matches_2[id].queryIdx].pt
        p2 = points_4[matches_2[id].trainIdx].pt
        cv.line(img_pair_2_aff, (int(p1[0]),int(p1[1])), (int(p2[0]+img_4.shape[1]),int(p2[1])), (0,0,255),1)

    cv.imshow("",img_pair_2_aff)
    cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-c-1.png", img_pair_2_aff)

    ## 3-d
    # apply inverse similarity transformation to simB to obtain simA
    print("applying similarity transform to simA")
    img_3_sim = img_3.copy()
    # cv.imshow("",img_3_sim)
    # cv.waitKey(0)

    img_3_sim = cv.warpAffine(img_3_sim, S_sim, (img_3_sim.shape[1], img_3_sim.shape[0]))

    # cv.imshow("",img_3_sim)
    # cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-d-1.png", img_3_sim)

    # plot simB warped and simA to R,B channels 
    img_3_4_sim = cv.merge([img_3_sim, np.zeros(img_3_sim.shape).astype("uint8"), img_4])

    cv.imshow("",img_3_4_sim)
    cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-d-2.png", img_3_4_sim)

    ## 3-e
    # apply inverse affine transformation to simB to obtain simA
    print("applying inverse affine transform to simb")
    img_3_aff = img_3.copy()
    # cv.imshow("",img_3_aff)
    # cv.waitKey(0)

    img_3_aff = cv.warpAffine(img_3_aff, S_aff, (img_3_aff.shape[1], img_3_aff.shape[0]))

    # cv.imshow("",img_3_aff)
    # cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-e-1.png", img_3_aff)

    img_3_4_aff = cv.merge([img_3_aff, np.zeros(img_3_aff.shape).astype("uint8"), img_4])

    cv.imshow("",img_3_4_aff)
    cv.waitKey(0)

    cv.imwrite("problem-sets/ps4/output/ps4-3-e-2.png", img_3_4_aff)
