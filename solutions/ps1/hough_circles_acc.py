# function H = hough_circles_acc(BW, radius)
#     % Compute Hough accumulator array for finding circles.
#     %
#     % BW: Binary (black and white) image containing edge pixels
#     % radius: Radius of circles to look for, in pixels

#     % TODO: Your code here
# end

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def hough_circles_acc(img_edge, max_radius, min_radius=15):
    # get x,y coords of edge pixels
    edge_pix = np.squeeze(np.where(np.max(img_edge) == img_edge)).T

    # define bins size for a,b
    bins_a = 2 * (img_edge.shape[1] - max_radius + 1)
    bins_b = 2 * (img_edge.shape[0] - max_radius + 1)

    bins_c = max_radius + 1

    bins_array = np.zeros((bins_a, bins_b, bins_c))
    
    for pix in edge_pix:
        y = pix[1]
        x = pix[0]

        for theta in range(0, 360, 1):
            for radius in range(min_radius, max_radius + 1):
                a = x - (radius * np.cos(np.deg2rad(theta)))
                b = y + (radius * np.sin(np.deg2rad(theta)))
                
                bins_array[int(a)][int(b)][int(radius)] += 1

    # get subsection of a,b space and transpose array
    bins_array = bins_array[0:img_edge.shape[0], 0:img_edge.shape[1], :].copy()
    bins_array = np.moveaxis(bins_array, 1, 0) 

    return bins_array, bins_a, bins_b, bins_c