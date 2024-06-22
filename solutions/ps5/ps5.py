from ps5_functions import *
import cv2 as cv
import numpy as np

from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def ps5_1_a():
    img_1 = cv.imread("solutions/ps5/input/TestSeq/Shift0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/TestSeq/ShiftR2.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/TestSeq/ShiftR5U5.png", cv.IMREAD_GRAYSCALE)

    # compute lk flow
    flow_12 = lk_flow(img_1, img_2, gauss_kern=-1, gauss_sigma=9, win_size=31)
    flow_13 = lk_flow(img_1, img_3, gauss_kern=-1, gauss_sigma=9, win_size=31)

    vis_optic_flow_arrows(img_1, flow_12, 'solutions/ps5/output/ps5-1-a-1.png', show=False)
    vis_optic_flow_arrows(img_1, flow_13, 'solutions/ps5/output/ps5-1-a-2.png', show=False)

    """
    TEXTUAL RESPONSE:
    I had to weight the kernel with a gaussian and increase the size of the window to avoid the "aperture" effect.
    """

def ps5_1_b():
    img_1 = cv.imread("solutions/ps5/input/TestSeq/Shift0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/TestSeq/ShiftR10.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/TestSeq/ShiftR20.png", cv.IMREAD_GRAYSCALE)
    img_4 = cv.imread("solutions/ps5/input/TestSeq/ShiftR40.png", cv.IMREAD_GRAYSCALE)

    # compute lk flow
    flow_12 = lk_flow(img_1, img_2, gauss_kern=-1, gauss_sigma=9, win_size=31)
    flow_13 = lk_flow(img_1, img_3, gauss_kern=-1, gauss_sigma=9, win_size=31)
    flow_14 = lk_flow(img_1, img_4, gauss_kern=-1, gauss_sigma=9, win_size=31)

    vis_optic_flow_arrows(img_1, flow_12, 'solutions/ps5/output/ps5-1-b-1.png', show=False)
    vis_optic_flow_arrows(img_1, flow_13, 'solutions/ps5/output/ps5-1-b-2.png', show=False)
    vis_optic_flow_arrows(img_1, flow_14, 'solutions/ps5/output/ps5-1-b-3.png', show=False)

    """
    TEXTUAL RESPONSE:
    The algorithm falls apart the bigger the displacement.This is because LK cannot handle larger displacements.
    """

def ps5_2_a():
    img_1 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_01.jpg", cv.IMREAD_GRAYSCALE)

    # apply downsampling
    gp = reduce_expand(img_1)

    img_gp = np.hstack((gp))
    cv.imwrite("solutions/ps5/output/ps5-2-a-1.png", img_gp)

def ps5_2_b():
    img_1 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_01.jpg", cv.IMREAD_GRAYSCALE)

    # apply downsampling
    gp = reduce_expand(img_1)

    # compute laplacian pyramid
    lp = laplacian_pyramid(gp)

    img_lp = np.hstack((lp))
    cv.imwrite("solutions/ps5/output/ps5-2-b-1.png", img_lp)

def ps5_3_a():
    # read images
    img_1 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_01.jpg", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_02.jpg", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_03.jpg", cv.IMREAD_GRAYSCALE)

    # build gaussian pyramids
    gp_1 = reduce_expand(img_1)
    gp_2 = reduce_expand(img_2)
    gp_3 = reduce_expand(img_3)

    # calculate optical flow
    flow_12 = lk_flow(gp_1[2], gp_2[2], gauss_kern=-1, gauss_sigma=15, win_size=31).astype("float32")
    flow_23 = lk_flow(gp_2[2], gp_3[2], gauss_kern=-1, gauss_sigma=15, win_size=31).astype("float32")
    # flow_12 = -lk_optic_flow(gp_1[0], gp_2[0], win=15)
    # flow_23 = -lk_optic_flow(gp_2[0], gp_3[0], win=15)

    # put flow images side by side
    img_12 = np.hstack((img_1, img_2))
    flow_13 = np.hstack((flow_12, flow_23))
    vis_optic_flow_arrows(img_12, flow_13, 'solutions/ps5/output/ps5-3-a-1.png', show=False, step_denom=20)

    # warp image 2 to image 1 using flow_1
    print(np.min(flow_12[:,:,0]), np.max(flow_12[:,:,0]),np.min(flow_12[:,:,1]), np.max(flow_12[:,:,1]))
    # print(np.min(flow_12_mine[:,:,0]), np.max(flow_12_mine[:,:,0]),np.min(flow_12_mine[:,:,1]), np.max(flow_12_mine[:,:,1]))
    # print(np.min(flow_23[:,:,0]), np.max(flow_23[:,:,0]),np.min(flow_23[:,:,1]), np.max(flow_23[:,:,1]))

    # build flow map
    flow_map_12 = np.zeros(flow_12.shape).astype("float32")
    print(flow_12.shape, img_1.shape, flow_12.shape)

    # increase magnitude
    MAGN = 1

    for i in range(flow_map_12.shape[0]):
        for j in range(flow_map_12.shape[1]):
            flow_map_12[i,j,0] = i + (MAGN * flow_12[i,j,0]) # map x
            flow_map_12[i,j,1] = j + (MAGN * flow_12[i,j,1]) # map y

    img_2_warp = cv.remap(img_2, flow_map_12[:,:,1], flow_map_12[:,:,0], cv.INTER_LINEAR)
    img_2_backwarp = backwarp(img_2, flow_12)

    img_1_diff = cv.subtract(img_1, img_2_backwarp)
    img_21_diff = cv.subtract(img_1, img_2)

    img_21 = np.hstack((img_1, img_2, img_2_backwarp, img_1_diff,img_21_diff))
    # cv.imshow("",img_21); cv.waitKey(0)
    cv.imwrite("solutions/ps5/output/ps5-3-a-2.png", img_1_diff)

    # dataseq 2
    # read images
    img_1 = cv.imread("solutions/ps5/input/DataSeq2/0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/DataSeq2/1.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/DataSeq2/2.png", cv.IMREAD_GRAYSCALE)

    # build gaussian pyramids
    gp_1 = reduce_expand(img_1)
    gp_2 = reduce_expand(img_2)
    gp_3 = reduce_expand(img_3)

    # calculate optical flow
    flow_12 = lk_flow(gp_1[1], gp_2[1], gauss_kern=-1, gauss_sigma=15, win_size=31).astype("float32")
    flow_23 = lk_flow(gp_2[1], gp_3[1], gauss_kern=-1, gauss_sigma=15, win_size=31).astype("float32")

    # put flow images side by side
    img_12 = np.hstack((img_1, img_2))
    flow_13 = np.hstack((flow_12, flow_23))
    vis_optic_flow_arrows(img_12, flow_13, 'solutions/ps5/output/ps5-3-a-3.png', show=False, step_denom=20)

    img_2_backwarp = backwarp(img_2, flow_12)
    img_1_diff = cv.subtract(img_1, img_2_backwarp)
    cv.imwrite("solutions/ps5/output/ps5-3-a-4.png", img_1_diff)


def ps5_4_a():
    img_1 = cv.imread("solutions/ps5/input/TestSeq/Shift0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/TestSeq/ShiftR10.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/TestSeq/ShiftR20.png", cv.IMREAD_GRAYSCALE)
    img_4 = cv.imread("solutions/ps5/input/TestSeq/ShiftR40.png", cv.IMREAD_GRAYSCALE)
    
    # compute flows
    flow_12 = lk_flow_iter(img_1, img_2, levels=4, win_size=5)
    flow_13 = lk_flow_iter(img_1, img_3, levels=4, win_size=5)
    flow_14 = lk_flow_iter(img_1, img_4, levels=5, win_size=5)
    
    # backwarp images
    img_2_backwarp = backwarp(img_2, -flow_12)
    img_3_backwarp = backwarp(img_3, -flow_13)
    img_4_backwarp = backwarp(img_4, -flow_14)

    # compute non warp diffs
    img_12_diff = cv.subtract(img_1, img_2)
    img_13_diff = cv.subtract(img_1, img_3)
    img_14_diff = cv.subtract(img_1, img_4)

    # compute warp diffs
    img_12_warp_diff = cv.subtract(img_1, img_2_backwarp)
    img_13_warp_diff = cv.subtract(img_1, img_3_backwarp)
    img_14_warp_diff = cv.subtract(img_1, img_4_backwarp)

    # vis optic flow pics
    img_1s = np.hstack((img_1, img_1, img_1))
    flows = np.hstack((flow_12, flow_13, flow_14))
    vis_optic_flow_arrows(img_1s, flows, 'solutions/ps5/output/ps5-4-a-1.png', show=False, step_denom=10)

    # vis image diffs
    img_diffs = np.hstack((img_12_diff, img_13_diff, img_14_diff))
    img_warp_diffs = np.hstack((img_12_warp_diff, img_13_warp_diff, img_14_warp_diff))
    img_diffs_comp = np.vstack((img_diffs, img_warp_diffs))

    cv.imwrite("solutions/ps5/output/ps5-4-a-2.png", img_warp_diffs)

def ps5_4_b():
    # read images
    img_1 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_01.jpg", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_02.jpg", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/DataSeq1/yos_img_03.jpg", cv.IMREAD_GRAYSCALE)

    # compute flows
    flow_12 = lk_flow_iter(img_1, img_2, levels=2, win_size=5)
    flow_13 = lk_flow_iter(img_1, img_3, levels=2, win_size=9)

    # backwarp images
    img_2_backwarp = backwarp(img_2, -flow_12)
    img_3_backwarp = backwarp(img_3, -flow_13)

    # compute non warp diffs
    img_12_diff = cv.subtract(img_1, img_2)
    img_13_diff = cv.subtract(img_1, img_3)

    # compute warp diffs
    img_12_warp_diff = cv.subtract(img_1, img_2_backwarp)
    img_13_warp_diff = cv.subtract(img_1, img_3_backwarp)

    # vis optic flow pics
    img_1s = np.hstack((img_1, img_1))
    flows = np.hstack((flow_12, flow_13))
    vis_optic_flow_arrows(img_1s, flows, 'solutions/ps5/output/ps5-4-b-1.png', show=False, step_denom=20)

    # vis image diffs
    img_diffs = np.hstack((img_12_diff, img_13_diff))
    img_warp_diffs = np.hstack((img_12_warp_diff, img_13_warp_diff))
    img_diffs_comp = np.vstack((img_diffs, img_warp_diffs))

    cv.imwrite("solutions/ps5/output/ps5-4-b-2.png", img_warp_diffs)

def ps5_4_c():
    # read images
    img_1 = cv.imread("solutions/ps5/input/DataSeq2/0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/DataSeq2/1.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/DataSeq2/2.png", cv.IMREAD_GRAYSCALE)

    # compute flows
    flow_12 = lk_flow_iter(img_1, img_2, levels=2, win_size=9)
    flow_13 = lk_flow_iter(img_1, img_3, levels=3, win_size=5)

    # backwarp images
    img_2_backwarp = backwarp(img_2, -flow_12)
    img_3_backwarp = backwarp(img_3, -flow_13)

    # compute non warp diffs
    img_12_diff = cv.subtract(img_1, img_2)
    img_13_diff = cv.subtract(img_1, img_3)

    # compute warp diffs
    img_12_warp_diff = cv.subtract(img_1, img_2_backwarp)
    img_13_warp_diff = cv.subtract(img_1, img_3_backwarp)

    # vis optic flow pics
    img_1s = np.hstack((img_1, img_1))
    flows = np.hstack((flow_12, flow_13))
    vis_optic_flow_arrows(img_1s, flows, 'solutions/ps5/output/ps5-4-c-1.png', show=False, step_denom=20)

    # vis image diffs
    img_diffs = np.hstack((img_12_diff, img_13_diff))
    img_warp_diffs = np.hstack((img_12_warp_diff, img_13_warp_diff))
    img_diffs_comp = np.vstack((img_diffs, img_warp_diffs))

    cv.imwrite("solutions/ps5/output/ps5-4-c-2.png", img_warp_diffs)

def ps5_5_a():
    # read images
    img_1 = cv.imread("solutions/ps5/input/Juggle/0.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("solutions/ps5/input/Juggle/1.png", cv.IMREAD_GRAYSCALE)
    img_3 = cv.imread("solutions/ps5/input/Juggle/2.png", cv.IMREAD_GRAYSCALE)

    # compute flows
    flow_12 = lk_flow_iter(img_1, img_2, levels=3, win_size=5)
    flow_13 = lk_flow_iter(img_1, img_3, levels=3, win_size=5)

    # backwarp images
    img_2_backwarp = backwarp(img_2, -flow_12)
    img_3_backwarp = backwarp(img_3, -flow_13)

    # compute non warp diffs
    img_12_diff = cv.subtract(img_1, img_2)
    img_13_diff = cv.subtract(img_1, img_3)

    # compute warp diffs
    img_12_warp_diff = cv.subtract(img_1, img_2_backwarp)
    img_13_warp_diff = cv.subtract(img_1, img_3_backwarp)

    # vis optic flow pics
    img_1s = np.hstack((img_1, img_1))
    flows = np.hstack((flow_12, flow_13))
    vis_optic_flow_arrows(img_1s, flows, 'solutions/ps5/output/ps5-5-a-1.png', show=False, step_denom=20)

    # vis image diffs
    img_diffs = np.hstack((img_12_diff, img_13_diff))
    img_warp_diffs = np.hstack((img_12_warp_diff, img_13_warp_diff))
    img_diffs_comp = np.vstack((img_diffs, img_warp_diffs))

    cv.imwrite("solutions/ps5/output/ps5-5-a-2.png", img_warp_diffs)

ps5_list = OrderedDict([('1a', ps5_1_a), ('1b', ps5_1_b), ('2a', ps5_2_a), ('2b', ps5_2_b), ('3a', ps5_3_a), ('4a', ps5_4_a), ('4b', ps5_4_b), ('4c', ps5_4_c), ('5a', ps5_5_a)])

if __name__=="__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] in ps5_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps5_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,1b,2a,2b,3a,4a,4b,4c,5a} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps5_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps5_list[idx]()