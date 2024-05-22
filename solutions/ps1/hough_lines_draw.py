# function hough_lines_draw(img, outfile, peaks, rho, theta)
#     % Draw lines found in an image using Hough transform.
#     %
#     % img: Image on top of which to draw lines
#     % outfile: Output image filename to save plot as
#     % peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
#     % rho: Vector of rho values, in pixels
#     % theta: Vector of theta values, in degrees

#     % TODO: Your code here
# end
import cv2 as cv
import numpy as np 

def hough_lines_draw(img, outfile, peaks, rho_list, theta_list):

    for peak in peaks:
        rho = peak[0] - len(rho_list)/2 
        theta = peak[1] * (np.pi/180)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x_0 = round(x0 + 1000*(-b))
        y_0 = round(y0 + 1000*(a))
        x_1 = round(x0 - 1000*(-b))
        y_1 = round(y0 - 1000*(a))

        cv.line(img, (x_0, y_0), (x_1, y_1), color=(255,255,0) ,thickness=2)

    cv.imwrite(outfile, img.astype("uint8"))