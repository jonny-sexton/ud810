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

def hough_peaks(H, n):
    peaks=[]
    
    for i in range(n):
        max_value = np.where(H == np.max(H))
        # print(len(max_value))

        if len(max_value) == 2:
            x = max_value[0][0]
            y = max_value[1][0]

            if H[x][y] == 0:
                break

            peaks.append([y,180 - x]) # don't know why I need to subtract it from 180°, but I do
            H[x][y] = 0
        
        elif len(max_value) == 3:
            x = max_value[0][0]
            y = max_value[1][0]
            z = max_value[2][0]

            if H[x][y][z] == 0:
                break

            peaks.append([y,180 - x, z]) # don't know why I need to subtract it from 180°, but I do
            H[x][y][z] = 0

    return np.array(peaks)