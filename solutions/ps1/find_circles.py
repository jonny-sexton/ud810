# function [centers, radii] = find_circles(BW, radius_range)
#     % Find circles in given radius range using Hough transform.
#     %
#     % BW: Binary (black and white) image containing edge pixels
#     % radius_range: Range of circle radii [min max] to look for, in pixels

#     % TODO: Your code here
# end

import numpy as np
import cv2 as cv

def find_circles(img, outfile, peaks, a, b, c):
    # get peaks and draw circle at peaks
    for peak in peaks:
        cv.circle(img, (180 - peak[1], peak[0]), peak[2], color=(0,0,255) ,thickness=2)

    cv.imwrite(outfile, img.astype("uint8"))