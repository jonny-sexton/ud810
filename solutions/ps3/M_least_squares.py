import numpy as np

def M_least_squares(pts_2d, pts_3d):
    # get number of points
    num_lines_2d = len(pts_2d)
    num_lines_3d = len(pts_3d)

    if num_lines_2d != num_lines_3d:
        print("Error: Num of 2d points does not match number of 3d points!")
        return None
    
    # create 2n * 12 matrix (A) and X
    A = np.zeros((2*num_lines_2d, 11))
    b = np.zeros((2*num_lines_2d, 1))

    # populate matrices
    for i in range(num_lines_2d):
        u, v = pts_2d[i]
        x, y, z = pts_3d[i]
        row_n = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z]
        row_n1 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]
        A[2*i] = row_n
        A[(2*i)+1] = row_n1
        b[2*i] = u
        b[(2*i)+1] = v

    # compute least squares
    M, res, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # append m23 and reshape M
    M = np.append(M, 1)
    M = np.reshape(M, (3,4))

    return M





