import cv2 as cv
import numpy as np

def F_least_squares(pts_2d_a, pts_2d_b):
    # get number of points
    num_lines_2d_a = len(pts_2d_a)
    num_lines_2d_b = len(pts_2d_b)

    if num_lines_2d_a != num_lines_2d_b:
        print("Error: Num of A points does not match number of B points!")
        return None
    
    # create 2n * 12 matrix (A) and X
    A = np.zeros((num_lines_2d_a, 8))
    b = -np.ones((num_lines_2d_a, 1))

    # populate matrices
    for i in range(num_lines_2d_a):
        u_a, v_a = pts_2d_a[i]
        u_b, v_b = pts_2d_b[i]
        row = [u_b*u_a, u_b*v_a, u_b, v_b*u_a, v_b*v_a, v_b, u_a, v_a]
        A[i] = row

    # # compute least squares
    F, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # append f33 and reshape F
    F = np.append(F, 1)
    F = np.reshape(F, (3,3))

    return F