# function [H, theta, rho] = hough_lines_acc(BW, varargin)
#      Compute Hough accumulator array for finding lines.
#     
#      BW: Binary (black and white) image containing edge pixels
#      RhoResolution (optional): Difference between successive rho values, in pixels
#      Theta (optional): Vector of theta values to use, in degrees
#     
#      Please see the Matlab documentation for hough():
#      http://www.mathworks.com/help/images/ref/hough.html
#      Your code should imitate the Matlab implementation.
#     
#      Pay close attention to the coordinate system specified in the assignment.
#      Note: Rows of H should correspond to values of rho, columns those of theta.

#      Parse input arguments
#     p = inputParser();
#     addParameter(p, 'RhoResolution', 1);
#     addParameter(p, 'Theta', linspace(-90, 89, 180));
#     parse(p, varargin{:});

#     rhoStep = p.Results.RhoResolution;
#     theta = p.Results.Theta;

#      TODO: Your code here
# end

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hough_lines_acc(img_edge, Theta=[0, 1, 180], RhoResolution=1):

    edge_pix = np.squeeze(np.where(np.max(img_edge) == img_edge)).T

    
    bins_size = int((Theta[2]-Theta[0])/Theta[1])
    bins_theta = np.linspace(Theta[0], Theta[2], bins_size+1)

    max_rho = round((img_edge.shape[0] + img_edge.shape[1]) / RhoResolution) 
    bins_rho = np.linspace(0,(max_rho*2), (max_rho*2)+1)

    bins_array = np.zeros((len(bins_theta), len(bins_rho)))

    for pix in edge_pix:
        x, y = pix

        for theta in range(-90, 90, 1): # don't know why it needs to be -90:90 instead of 0:180, but it does.
            rho = round((x * np.cos(np.deg2rad(theta))) + (y * np.sin(np.deg2rad(theta))))

            # vote
            bins_array[theta+90][rho+max_rho] += 1

    # plt.imshow(bins_array.T, interpolation='none', cmap="RdYlBu")
    # plt.show()

    return(bins_array, bins_theta, bins_rho)