import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel, generic_gradient_magnitude
import math
import matplotlib.pyplot as plt

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

def lk_flow(img_t, img_t1, gauss_kern=-1, gauss_sigma=1,  win_size=5):
    # convert image to float64 and normalize
    # img_t = normalize_01(img_t.astype("float64"))
    # img_t1 = normalize_01(img_t1.astype("float64"))
    
    # check if images are same size
    if img_t.shape != img_t1.shape:
        print("images must have the same dimensions!")
        return None
    
    # compute window mid coord
    win_mid = (win_size // 2 ) + 1

    # normalize images to [0,1]
    img_t = normalize_01(img_t)
    img_t1 = normalize_01(img_t1)

    # compute I_x, I_y, I_t
    I_x_orig = img_grad(img_t, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=0)
    I_y_orig = img_grad(img_t, gauss_kern=gauss_kern, gauss_sigma=gauss_sigma, axis=1)
    I_t_orig = cv.subtract(img_t, img_t1)
    I_x = np.zeros(img_t.shape, dtype=np.float32)
    I_y = np.zeros(img_t.shape, dtype=np.float32)
    I_t = np.zeros(img_t.shape, dtype=np.float32)
    I_x[1:-1, 1:-1] = cv.subtract(img_t[1:-1, 2:], img_t[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = cv.subtract(img_t[2:, 1:-1], img_t[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = cv.subtract(img_t[1:-1, 1:-1], img_t1[1:-1, 1:-1])
    gkern = gaussian_kernel(l=win_size, sig=win_size)

    uv = np.zeros((img_t.shape[0],img_t.shape[1],2))

    # print(np.min(img_t), np.max(img_t))
    # print(np.min(I_x), np.max(I_x))
    # print(np.min(I_x_orig), np.max(I_x_orig))

    # I_x_s = np.hstack((I_x, I_x_orig))
    # cv.imshow("", I_x_s); cv.waitKey(0)

    # iterate each pixel with window size (u, z)
    for i in range(img_t.shape[0]):
        for j in range(img_t.shape[1]):
            # get patch around pixel from deriv images
            patch_x = I_x[max(0,i-win_mid+1):min(img_t.shape[0],i+win_mid), max(0,j-win_mid+1):min(img_t.shape[1],j+win_mid)]
            patch_y = I_y[max(0,i-win_mid+1):min(img_t.shape[0],i+win_mid), max(0,j-win_mid+1):min(img_t.shape[1],j+win_mid)]
            patch_t = I_t[max(0,i-win_mid+1):min(img_t.shape[0],i+win_mid), max(0,j-win_mid+1):min(img_t.shape[1],j+win_mid)]
            patch_gkern = gkern[max(0,win_mid-i-1):min(win_size,img_t.shape[0]-i+win_mid-1),max(0,win_mid-j-1):min(win_size,img_t.shape[1]-j+win_mid-1)]

            # apply gaussian weighted average to patches
            patch_x_conv = patch_x * patch_gkern
            patch_y_conv = patch_y * patch_gkern
            patch_t_conv = patch_t * patch_gkern

            # build matrixes
            A = M(patch_x, patch_y)
            b = np.array([[-np.sum(patch_x*patch_t)],[-np.sum(patch_y*patch_t)]])
            # A = M(patch_x_conv, patch_y_conv)
            # b = np.array([[-np.sum(patch_x_conv*patch_t_conv)],[-np.sum(patch_y_conv*patch_t_conv)]])

            # solve least squares
            flow = np.linalg.lstsq(A, b, rcond=None)[0]

            uv[i,j,0] = flow[0]
            uv[i,j,1] = flow[1]

    return uv

def vis_optic_flow_arrows(img, flow, filename, show=True, step_denom=50):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = img.shape[0] // step_denom
    plt.quiver(x[::step, ::step], y[::step, ::step],
               flow[::step, ::step, 0], flow[::step, ::step, 1],
               color='r', pivot='middle', headwidth=2, headlength=3)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()

def reduce(img, levels=4):
    # TODO: img_bd = ? (blur by 5x5 gaussian, then downsample)
    img_size = img.shape
    imgs = [img]
    for i in range(levels-1):
        img = cv.GaussianBlur(img, (5,5), 0)
        img = cv.resize(img, # original image
                        (0,0), # set fx and fy, not the final size
                        fx=0.5, 
                        fy=0.5, 
                        interpolation=cv.INTER_NEAREST)
        imgs.append(img)

    return imgs

def expand(flow):
    return cv.resize(flow, # original image
                        None, # set fx and fy, not the final size
                        fx=2,
                        fy=2,
                        interpolation=cv.INTER_LINEAR)

def reduce_expand(img, levels=4):
    # TODO: img_bd = ? (blur by 5x5 gaussian, then downsample)
    img_size = img.shape
    imgs = [img]
    for i in range(levels-1):
        img = cv.GaussianBlur(img, (5,5), 0)
        img = cv.resize(img, # original image
                        None, # set fx and fy, not the final size
                        fx=0.5, 
                        fy=0.5, 
                        interpolation=cv.INTER_NEAREST)
        imgs.append(img)

    for i in range(levels):
        imgs[i] = cv.resize(imgs[i], # original image
                        (img_size[1],img_size[0]), # set fx and fy, not the final size
                        fx=0,
                        fy=0,
                        interpolation=cv.INTER_LINEAR)

    return imgs

def laplacian_pyramid(gp):
    lp = [gp[-1]]
    for i in range(len(gp)-1):
        # get images
        img_t = gp[-i-1]
        img_t1 = gp[-i-2]

        # compute laplacian
        img_t = normalize_01(img_t)
        img_t1 = normalize_01(img_t1)
        img_lp = img_t - img_t1

        img_lp = normalize_0255(img_lp).astype("uint8")
        img_t = normalize_0255(img_t).astype("uint8")
        img_t1 = normalize_0255(img_t1).astype("uint8")

        # prepend to list
        lp.insert(0,img_lp)

    return lp

def lk_optic_flow(frame1, frame2, win=2):
    '''
    The code below was borrowed from stackoverflow
    ../questions/14321092/lucas-kanade-python-numpy-implementation-uses-enormous-amount-of-memory
    '''

    # calculate gradients in x, y and t dimensions
    Ix = np.zeros(frame1.shape, dtype=np.float32)
    Iy = np.zeros(frame1.shape, dtype=np.float32)
    It = np.zeros(frame1.shape, dtype=np.float32)
    Ix[1:-1, 1:-1] = cv.subtract(frame1[1:-1, 2:], frame1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = cv.subtract(frame1[2:, 1:-1], frame1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = cv.subtract(frame1[1:-1, 1:-1], frame2[1:-1, 1:-1])

    params = np.zeros(frame1.shape + (5,))
    params[..., 0] = Ix ** 2
    params[..., 1] = Iy ** 2
    params[..., 2] = Ix * Iy
    params[..., 3] = Ix * It
    params[..., 4] = Iy * It
    del It, Ix, Iy
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(frame1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2

    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    op_flow = op_flow.astype(np.float32)
    return op_flow

def backwarp(img, flow):
    h, w = flow.shape[:2]
    flow_map = -flow.copy()
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    warped = cv.remap(img, flow_map, None, cv.INTER_LINEAR)
    return warped

def lk_flow_iter(img_t, img_t1, levels=3, win_size=31):
    # compute gaussian pyramid
    gp_t = reduce(img_t, levels=levels)
    gp_t1 = reduce(img_t1, levels=levels)

    # perform LK on level k and obtain flow field
    for i in range(levels-1):
        level = levels - i - 1

        if i == 0:
            flow = lk_optic_flow(gp_t[level], gp_t1[level], win=win_size).astype("float32")
        else:
            flow = flow_iter

        # expand flow field and multiply values by 2x
        flow = 2 * expand(flow)

        # warp img_t @ level k-1 to obtain img_t'_k-1
        warped = backwarp(gp_t[level-1], flow)

        # perform LK on img_t'_k-1 to obtain "correction" flow field
        flow_corr = lk_optic_flow(warped, gp_t1[level-1], win=win_size).astype("float32")

        # add correction flow field to obtained flow field
        flow_iter = flow + flow_corr
    
    return flow_iter