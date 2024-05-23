# function peaks = hough_peaks(H, varargin)
#      Find peaks in a Hough accumulator array.
#     
#      Threshold (optional): Threshold at which values of H are considered to be peaks
#      NHoodSize (optional): Size of the suppression neighborhood, [M N]
#     
#      Please see the Matlab documentation for houghpeaks():
#      http://www.mathworks.com/help/images/ref/houghpeaks.html
#      Your code should imitate the matlab implementation.

#      Parse input arguments
#     p = inputParser;
#     addOptional(p, 'numpeaks', 1, @isnumeric);
#     addParameter(p, 'Threshold', 0.5 * max(H(:)));
#     addParameter(p, 'NHoodSize', floor(size(H) / 100.0) * 2 + 1);   odd values >= size(H)/50
#     parse(p, varargin{:});

#     numpeaks = p.Results.numpeaks;
#     threshold = p.Results.Threshold;
#     nHoodSize = p.Results.NHoodSize;

#      TODO: Your code here
# end

import numpy as np
import cv2 as cv

import sys
sys.path.append('./solutions/utils/')
from toolbox import *

def hough_peaks(H, n, nhood_size=10):
    peaks=[]
    
    for i in range(n):
        max_value = np.where(H == np.max(H))

        if len(max_value) == 2:
            x = max_value[0][0]
            y = max_value[1][0]

            if H[x,y] == 0:
                break

            peaks.append([y,180 - x]) # don't know why I need to subtract it from 180°, but I do
            x_range_min = int(max(0, x-nhood_size))
            x_range_max = int(min(H.shape[0], x+nhood_size))
            y_range_min = int(max(0, y-nhood_size))
            y_range_max = int(min(H.shape[1], y+nhood_size))

            # nms
            H[x_range_min:x_range_max,y_range_min:y_range_max] = 0
        
        elif len(max_value) == 3:
            x = max_value[0][0]
            y = max_value[1][0]
            r = max_value[2][0]

            if H[x][y][r] == 0:
                break

            peaks.append([y,180 - x, r]) # don't know why I need to subtract it from 180°, but I do
            x_range_min = int(max(0, x-nhood_size))
            x_range_max = int(min(H.shape[0], x+nhood_size))
            y_range_min = int(max(0, y-nhood_size))
            y_range_max = int(min(H.shape[1], y+nhood_size))
            r_range_min = int(max(0, r-nhood_size))
            r_range_max = int(min(H.shape[1], r+nhood_size))

            # nms
            H[x_range_min:x_range_max,y_range_min:y_range_max,r_range_min:r_range_max] = 0

    return np.array(peaks)