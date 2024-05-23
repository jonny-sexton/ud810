import numpy as np

def M_SVD(pts_2d, pts_3d):
    # get number of points
    num_lines_2d = len(pts_2d)
    num_lines_3d = len(pts_3d)

    if num_lines_2d != num_lines_3d:
        print("Error: Num of 2d points does not match number of 3d points!")
        return None
    
    # create 2n * 12 matrix (A) and X
    A = np.zeros((2*num_lines_2d, 12))

    # populate matrices
    for i in range(num_lines_2d):
        u, v = pts_2d[i]
        x, y, z = pts_3d[i]
        row_n = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
        row_n1 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
        A[2*i] = row_n
        A[(2*i)+1] = row_n1

    # get svd of A
    U, D, V_T = np.linalg.svd(A)

    # get last column of V
    V = V_T.T
    M = V[:,-1]

    # reshape M
    M = np.reshape(M, (3,4))

    return M





