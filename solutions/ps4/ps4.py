import cv2 as cv
import numpy as np
from ps4_functions import img_grad, M, C, harris_corners, thresh, nms, keypoint_angle, ransac_trans, ransac_sim, ransac_aff
import math

from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def ps4_1_a():
    img = normalize_01(cv.imread("solutions/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE))
    img_x = normalize_0255(img_grad(img, gauss_kern=1, axis=0)).astype("uint8")
    img_y = normalize_0255(img_grad(img, gauss_kern=1, axis=1)).astype("uint8")
    img_xy = np.hstack((img_x,img_y))

    # cv.imshow('', img_xy); cv.waitKey(0); cv.destroyAllWindows()
    cv.imwrite("solutions/ps4/output/ps4-1-a-1.png", img_xy)

    img = normalize_01(cv.imread("solutions/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE))
    img_x = normalize_0255(img_grad(img, gauss_kern=1, axis=0)).astype("uint8")
    img_y = normalize_0255(img_grad(img, gauss_kern=1, axis=1)).astype("uint8")
    img_xy = np.hstack((img_x,img_y))

    # cv.imshow('', img_xy); cv.waitKey(0); cv.destroyAllWindows()
    cv.imwrite("solutions/ps4/output/ps4-1-a-2.png", img_xy)

def ps4_1_b():
    img_1 = cv.imread("solutions/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_1 = harris_corners(img_1.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    cv.imwrite("solutions/ps4/output/ps4-1-b-1.png", R_1)

    img_2 = cv.imread("solutions/ps4/input/transB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_2 = harris_corners(img_2.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    cv.imwrite("solutions/ps4/output/ps4-1-b-2.png", R_2)

    img_3 = cv.imread("solutions/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_3 = harris_corners(img_3.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    cv.imwrite("solutions/ps4/output/ps4-1-b-3.png", R_3)

    img_4 = cv.imread("solutions/ps4/input/simB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_4 = harris_corners(img_4.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)
    cv.imwrite("solutions/ps4/output/ps4-1-b-4.png", R_4)

def ps4_1_c():
    img_1 = cv.imread("solutions/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_1 = harris_corners(img_1.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)

    img_2 = cv.imread("solutions/ps4/input/transB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_2 = harris_corners(img_2.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)

    img_3 = cv.imread("solutions/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    _, R_3 = harris_corners(img_3.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)

    img_4 = cv.imread("solutions/ps4/input/simB.jpg", cv.IMREAD_GRAYSCALE)
    _, R_4 = harris_corners(img_4.astype("float64").copy(), gauss_kern=3, gauss_sigma=3, win_size=3, sigma=1, alpha=0.04)

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
    img_1 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR) # convert to bgr
    img_2 = cv.cvtColor(img_2, cv.COLOR_GRAY2BGR) # convert to bgr
    img_3 = cv.cvtColor(img_3, cv.COLOR_GRAY2BGR) # convert to bgr
    img_4 = cv.cvtColor(img_4, cv.COLOR_GRAY2BGR) # convert to bgr

    global corners_1, corners_2, corners_3, corners_4
    corners_1 = np.nonzero(R_1)
    corners_2 = np.nonzero(R_2)
    corners_3 = np.nonzero(R_3)
    corners_4 = np.nonzero(R_4)

    for i in range(len(corners_1[0])):
        cv.circle(img_1, (corners_1[1][i], corners_1[0][i]), 2, (0, 0, 255), 2)
    for i in range(len(corners_2[0])):
        cv.circle(img_2, (corners_2[1][i], corners_2[0][i]), 2, (0, 0, 255), 2)
    for i in range(len(corners_3[0])):
        cv.circle(img_3, (corners_3[1][i], corners_3[0][i]), 2, (0, 0, 255), 2)
    for i in range(len(corners_4[0])):
        cv.circle(img_4, (corners_4[1][i], corners_4[0][i]), 2, (0, 0, 255), 2)

    cv.imwrite("solutions/ps4/output/ps4-1-c-1.png", img_1)
    cv.imwrite("solutions/ps4/output/ps4-1-c-2.png", img_2)
    cv.imwrite("solutions/ps4/output/ps4-1-c-3.png", img_3)
    cv.imwrite("solutions/ps4/output/ps4-1-c-4.png", img_4)

    """
    TEXT RESPONSE:
    - Setting the Sigma to 2 for computing the Harris value (sigma=2) made a huge difference. Then just need to fine tune the threshold to get as many features as possible.
    - Finding the maximum number of features from the Harris value map was improved massively by using the mean pixel intensity to compute the threshold value.
    - The rotated image does not behave so well. I imagine this is because the partial derivatives I_x, I_y become I_xy. This value is set to zero in the M matrix. Furthermore, due to sampling issues, the diagonal lines create a lot of small derivatives in the I_x, I_y direction.
    """

def ps4_2_a():
    img_1 = cv.imread("solutions/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps4/input/transB.jpg", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    img_4 = cv.imread("solutions/ps4/input/simB.jpg", cv.IMREAD_GRAYSCALE)

    print("computing keypoint angles...")
    global keypoints_1, keypoints_2, keypoints_3, keypoints_4
    keypoints_1 = keypoint_angle(img_1, corners_1)
    keypoints_2 = keypoint_angle(img_2, corners_2)
    keypoints_3 = keypoint_angle(img_3, corners_3)
    keypoints_4 = keypoint_angle(img_4, corners_4)
    
    # # draw keypoints
    img_1 = cv.drawKeypoints(img_1, keypoints_1, img_1, flags=4)
    img_2 = cv.drawKeypoints(img_2, keypoints_2, img_2, flags=4)
    img_3 = cv.drawKeypoints(img_3, keypoints_3, img_3, flags=4)
    img_4 = cv.drawKeypoints(img_4, keypoints_4, img_4, flags=4)

    cv.imwrite("solutions/ps4/output/ps4-2-a-1.png", np.hstack((img_1,img_2)))
    cv.imwrite("solutions/ps4/output/ps4-2-a-2.png", np.hstack((img_3,img_4)))

def ps4_2_b():
    global img_1, img_2, img_3, img_4
    img_1 = cv.imread("solutions/ps4/input/transA.jpg", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps4/input/transB.jpg", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps4/input/simA.jpg", cv.IMREAD_GRAYSCALE)
    img_4 = cv.imread("solutions/ps4/input/simB.jpg", cv.IMREAD_GRAYSCALE)
    
    # get descriptors
    print("computing descriptors...")
    sift = cv.SIFT_create()

    global points_1, points_2, points_3, points_4
    points_1, descriptors_1 = sift.compute(img_1,keypoints_1)
    points_2, descriptors_2 = sift.compute(img_2,keypoints_2)
    points_3, descriptors_3 = sift.compute(img_3,keypoints_3)
    points_4, descriptors_4 = sift.compute(img_4,keypoints_4)

    # compute matches
    print("computing matches...")
    bfm = cv.BFMatcher()
    global matches_1, matches_2
    matches_1 = bfm.match(descriptors_1, descriptors_2)
    matches_2 = bfm.match(descriptors_3, descriptors_4)
    
    # join images
    global img_pair_1, img_pair_2
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

    cv.imwrite("solutions/ps4/output/ps4-2-b-1.png", img_pair_matches_1)
    cv.imwrite("solutions/ps4/output/ps4-2-b-2.png", img_pair_matches_2)

def ps4_3_a():
    # compute ransac best transform
    consensus_set, S = ransac_trans(matches_1, points_1, points_2)

    for id in consensus_set:
        p1 = points_1[matches_1[id].queryIdx].pt
        p2 = points_2[matches_1[id].trainIdx].pt
        cv.line(img_pair_1, (int(p1[0]),int(p1[1])), (int(p2[0]+img_2.shape[1]),int(p2[1])), (0,0,255),1)

    print("S:\n",S)

    cv.imwrite("solutions/ps4/output/ps4-3-a-1.png", img_pair_1)

def ps4_3_b():
    # compute ransac similarity transform
    print("computing ransac similarity transform")
    global S_sim
    consensus_set_sim, S_sim = ransac_sim(matches_2, points_3, points_4, error=3)
    img_pair_2_sim = img_pair_2.copy()

    for id in consensus_set_sim:
        p1 = points_3[matches_2[id].queryIdx].pt
        p2 = points_4[matches_2[id].trainIdx].pt
        cv.line(img_pair_2_sim, (int(p1[0]),int(p1[1])), (int(p2[0]+img_4.shape[1]),int(p2[1])), (0,0,255),1)

    print("S_sim:\n",S_sim)

    cv.imwrite("solutions/ps4/output/ps4-3-b-1.png", img_pair_2_sim)

def ps4_3_c():
    # compute ransac affine transform
    print("computing ransac affine transform")
    global S_aff
    consensus_set_aff, S_aff = ransac_aff(matches_2, points_3, points_4, error=3)
    img_pair_2_aff = img_pair_2.copy()

    for id in consensus_set_aff:
        p1 = points_3[matches_2[id].queryIdx].pt
        p2 = points_4[matches_2[id].trainIdx].pt
        cv.line(img_pair_2_aff, (int(p1[0]),int(p1[1])), (int(p2[0]+img_4.shape[1]),int(p2[1])), (0,0,255),1)

    print("S_aff:\n",S_aff)

    cv.imwrite("solutions/ps4/output/ps4-3-c-1.png", img_pair_2_aff)

def ps4_3_d():
    # apply inverse similarity transformation to simB to obtain simA
    print("applying similarity transform to simA")
    img_3_sim = img_3.copy()
    img_3_sim = cv.warpAffine(img_3_sim, S_sim, (img_3_sim.shape[1], img_3_sim.shape[0]))
    cv.imwrite("solutions/ps4/output/ps4-3-d-1.png", img_3_sim)

    # plot simB warped and simA to R,B channels 
    img_3_4_sim = cv.merge([img_3_sim, np.zeros(img_3_sim.shape).astype("uint8"), img_4])
    cv.imwrite("solutions/ps4/output/ps4-3-d-2.png", img_3_4_sim)

def ps4_3_e():
    # apply inverse affine transformation to simB to obtain simA
    print("applying inverse affine transform to simb")
    img_3_aff = img_3.copy()
    img_3_aff = cv.warpAffine(img_3_aff, S_aff, (img_3_aff.shape[1], img_3_aff.shape[0]))
    cv.imwrite("solutions/ps4/output/ps4-3-e-1.png", img_3_aff)

    img_3_4_aff = cv.merge([img_3_aff, np.zeros(img_3_aff.shape).astype("uint8"), img_4])
    cv.imwrite("solutions/ps4/output/ps4-3-e-2.png", img_3_4_aff)    

ps4_list = OrderedDict([('1a', ps4_1_a), ('1b', ps4_1_b), ('1c', ps4_1_c), ('2a', ps4_2_a), ('2b', ps4_2_b), ('3a', ps4_3_a), ('3b', ps4_3_b), ('3c', ps4_3_c), ('3d', ps4_3_d), ('3e', ps4_3_e)])

if __name__=="__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] in ps4_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps4_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,1b,1c,2a,2b,3a,3b,3c,3d,3e} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps4_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps4_list[idx]()