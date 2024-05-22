import cv2 as cv
import numpy as np
from M_least_squares import M_least_squares
from F_least_squares import F_least_squares
from M_SVD import M_SVD
from F_SVD import F_SVD

def visualize_pts(img, pts): 
    
    for pt in pts:
        x = pt[0]
        y = pt[1]
        x = int(x) - 1
        y = int(y) - 1
        # print(x, y)
        img = cv.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    cv.imshow("", img)
    cv.waitKey(0)


if __name__ == "__main__":

    # visualize points
    img = cv.imread("problem-sets/ps3/input/pic_a.jpg")
    pts_img = "problem-sets/ps3/input/pts2d-pic_a.txt"
    #visualize_pts(img, pts)

    # read points files as list of points
    fp_pts_2d = "problem-sets/ps3/input/pts2d-norm-pic_a.txt"
    fp_pts_3d = "problem-sets/ps3/input/pts3d-norm.txt"

    pts_2d = []
    pts_3d = [] 

    # read files and append to lists
    with open(fp_pts_2d) as file_pts_2d, open(fp_pts_3d) as file_pts_3d:
        for line1, line2 in zip(file_pts_2d, file_pts_3d):
            u, v = list(map(float,line1.split()))
            x, y, z = list(map(float,line2.split()))

            pts_2d.append([u, v])
            pts_3d.append([x, y, z])
    
    # convert to numpy array
    pts_2d = np.array(pts_2d)
    pts_3d = np.array(pts_3d)

    # M = M_least_squares(pts_2d, pts_3d)
    M = M_SVD(pts_2d, pts_3d)

    ## 1-a
    print("1-a\n")
    # print M
    print("M:\n",M)

    # get 3d point
    pt_3d = pts_3d[-1]
    pt_3d = np.append(pt_3d, 1)
    print("Last <x,y,z> point:\n", pt_3d)

    # project to image
    pt_2d = M @ pt_3d

    # divide out by last term (homogeneous value) to get inhomogeneous point
    pt_2d /= pt_2d[2]
    print("<u,v> projection:\n", pt_2d)

    # compute residual
    res = (pt_2d[0] - pts_2d[-1][0], pt_2d[1] - pts_2d[-1][1])
    print("residual:\n", res)

    ## 2-a
    # get points
    # read points files as list of points
    fp_pts_2d = "problem-sets/ps3/input/pts2d-pic_a.txt"
    fp_pts_3d = "problem-sets/ps3/input/pts3d.txt"

    pts_2d = []
    pts_3d = [] 

    # read files and append to lists
    with open(fp_pts_2d) as file_pts_2d, open(fp_pts_3d) as file_pts_3d:
        for line1, line2 in zip(file_pts_2d, file_pts_3d):
            u, v = list(map(float,line1.split()))
            x, y, z = list(map(float,line2.split()))

            pts_2d.append([u, v])
            pts_3d.append([x, y, z])
    
    # convert to numpy array
    pts_2d = np.array(pts_2d)
    pts_3d = np.array(pts_3d)

    # avg k
    res_k_avg = {}

    # pick random points of size k = [8, 12, 16] 10 times
    set_sizes = [8, 12, 16]
    for i in range(10):
        res_dict = {}
        for k in set_sizes:
            idx_train = np.random.default_rng().choice(20, size=k, replace=False)
            idx_test = [x for x in list(range(20)) if x not in idx_train]

            pts_2d_rand_train = pts_2d[idx_train, :]
            pts_3d_rand_train = pts_3d[idx_train, :]

            pts_2d_rand_test_gt = pts_2d[idx_test, :]
            pts_3d_rand_test = pts_3d[idx_test, :]

            # compute M
            M = M_least_squares(pts_2d_rand_train, pts_3d_rand_train)
            # M = M_SVD(pts_2d_rand_train, pts_3d_rand_train)

            # print M
            # print("M:\n",M)

            # get first 4 random test 3d points and add homogenous column
            pts_3d_rand_test = pts_3d_rand_test[:4]
            pts_3d_rand_test = np.c_[pts_3d_rand_test, np.ones(4)]
            # print("Random <x,y,z> points:\n", pts_3d_rand_test)

            # res for k list
            res_k = [0.0, 0.0]
            
            # project to image
            # print("<u,v> projection:")
            for j in range(len(pts_3d_rand_test)):# pt_3d_rand_test in pts_3d_rand_test:
                pt_2d_random_test_proj = M @ pts_3d_rand_test[j]
                pt_2d_random_test_proj /= pt_2d_random_test_proj[-1]
                # print(pt_2d_random_test_proj)

                # compute residual
                res = (pt_2d_random_test_proj[0] - pts_2d_rand_test_gt[j][0], pt_2d_random_test_proj[1] - pts_2d_rand_test_gt[j][1])

                # store residual as ssd
                res_k[0] += res[0]**2
                res_k[1] += res[1]**2
            
            
            # print("Residual (SSD) for", k, "set size:", res_k)
            res_dict[k] = [res_k, M]

        # find best K and compute average K
        best_k = [-1, 0.0, 0]
        for k in res_dict:
            res = round(res_dict[k][0][0] + res_dict[k][0][1],2)
            M = res_dict[k][1]

            try: 
                res_k_avg[k] += (res / 10)
            except: 
                res_k_avg[k] = (res / 10)

            if best_k[0] == -1:
                best_k = [k, res, M]
            else:
                if best_k[1] > res:
                    best_k = [k, res, M]
        
        print("Best K:", best_k[0])
        print("Res:", best_k[1])
        print("M:", best_k[2])
        print()

    print("Avg Ks:")
    print(res_k_avg)


    """
    TEXT RESPONSE:
    Avg Ks:
    {8: 54.39699999999999, 12: 9.936000000000002, 16: 8.856}

    Explanation:
    K size 16 _usually_ produces the best result. I guess that sometimes a too narrow set of points (in a spatial sense) are selected, leading to a biased M matrix.

    Best M:
    K: 16
    Res: 1.21
    M: [[-2.33444568e+00 -1.08409788e-01  3.36167456e-01  7.36784208e+02]
    [-2.30936393e-01 -4.79481474e-01  2.08969314e+00  1.53505539e+02]
    [-1.26382400e-03 -2.06754310e-03  5.12567068e-04  1.00000000e+00]]
    """

    ## 1-c
    print("\n## 1-c")
    M = np.array([[-2.33444568, -0.10840978, 0.33616745,  736.784208],
    [-0.23093639, -0.47948147,  2.08969314,  153.505539],
    [-0.00126382, -0.00206754,  0.00051256,  1.0]])

    # M = best_k[2]
    # print(M)

    Q = M[:,:3]
    m_4 = M[:,3]


    C = -np.linalg.inv(Q) @ m_4

    print("C:", C) 

    ## 2-a
    # read points files as list of points
    fp_pts_2d_a = "problem-sets/ps3/input/pts2d-pic_a.txt"
    fp_pts_2d_b = "problem-sets/ps3/input/pts2d-pic_b.txt"

    pts_2d_a = []
    pts_2d_b = [] 

    # read files and append to lists
    with open(fp_pts_2d_a) as file_pts_2d_a, open(fp_pts_2d_b) as file_pts_2d_b:
        for line1, line2 in zip(file_pts_2d_a, file_pts_2d_b):
            u_a, v_a = list(map(float,line1.split()))
            u_b, v_b = list(map(float,line2.split()))

            pts_2d_a.append([u_a, v_a])
            pts_2d_b.append([u_b, v_b])
    
    # convert to numpy array
    pts_2d_a = np.array(pts_2d_a)
    pts_2d_b = np.array(pts_2d_b)

    # F = F_least_squares(pts_2d_a, pts_2d_b)
    F = F_SVD(pts_2d_a, pts_2d_b)

    print("\n## 2-a")
    print("F:", F)

    ## 2-b
    print("\n## 2-b")
    # compute SVD of F
    U, D, V_T = np.linalg.svd(F)

    D = np.array([[D[0], 0, 0], [0, D[1], 0], [0, 0, 0]])

    F_hat = U @ D @ V_T

    print("F_hat:", F_hat)

    ## 2-c
    print("## 2-c")

    pic_a = cv.imread("problem-sets/ps3/input/pic_a.jpg")
    pic_b = cv.imread("problem-sets/ps3/input/pic_b.jpg")

    pts_2d_a = np.column_stack((pts_2d_a, np.ones(pts_2d_a.shape[0])))
    pts_2d_b = np.column_stack((pts_2d_b, np.ones(pts_2d_b.shape[0])))

    F = F_hat
    
    epilines_a = pts_2d_b @ F
    epilines_b = pts_2d_a @ F.T

    l_l = np.cross([0, 0, 1], [pic_a.shape[0], 0, 1])
    l_r = np.cross([0, pic_a.shape[1], 1], [pic_a.shape[0], pic_a.shape[1], 1])

    for l_a, l_b in zip(epilines_a, epilines_b):
        p_a_l = np.cross(l_a, l_l)
        p_a_r = np.cross(l_a, l_r)
        p_a_l = (p_a_l / p_a_l[-1]).astype(int)
        p_a_r = (p_a_r / p_a_r[-1]).astype(int)
        cv.line(pic_a, (p_a_l[:2]), (p_a_r[:2]), (0, 0, 255), thickness=2)
        p_b_l = np.cross(l_b, l_l)
        p_b_r = np.cross(l_b, l_r)
        p_b_l = (p_b_l / p_b_l[-1]).astype(int)
        p_b_r = (p_b_r / p_b_r[-1]).astype(int)
        cv.line(pic_b, (p_b_l[:2]), (p_b_r[:2]), (0, 0, 255), thickness=2)

    cv.imshow('', np.hstack((pic_b,pic_a))); cv.waitKey(0); cv.destroyAllWindows()

    # cv.imwrite("problem-sets/ps3/output/ps3-2-c-1.png", pic_a)
    # cv.imwrite("problem-sets/ps3/output/ps3-2-c-2.png", pic_b)

    ## 2-d
    print("\n## 2-d")
    pic_a = cv.imread("problem-sets/ps3/input/pic_a.jpg")
    pic_b = cv.imread("problem-sets/ps3/input/pic_b.jpg")

    # shift centroid to average centre of points
    com_a = np.mean(pts_2d_a, axis=0)
    delta_a = np.array((0, 0)) - com_a[:2]

    com_b = np.mean(pts_2d_b, axis=0)
    delta_b = np.array((0, 0)) - com_b[:2]

    # build translation matrix T 
    T_a = np.array([[1, 0, delta_a[0]], [0, 1, delta_a[1]], [0, 0, 1]])
    T_b = np.array([[1, 0, delta_b[0]], [0, 1, delta_b[1]], [0, 0, 1]])

    # scale so average distance of points from centroid sqrt(2)
    theta_a = np.sqrt(2) / np.mean(pts_2d_a, axis=0)
    theta_b = np.sqrt(2) / np.mean(pts_2d_b, axis=0)

    # build scale matrix S
    S_a = np.array([[theta_a[0], 0, 0], [0, theta_a[1], 0], [0, 0, 1]])
    S_b = np.array([[theta_b[0], 0, 0], [0, theta_b[1], 0], [0, 0, 1]])

    # build affine transformation (translation and scale)
    A_a = S_a @ T_a
    A_b = S_b @ T_b

    # transform points
    pts_2d_a_aff = []
    for pt in pts_2d_a:
        pt = A_a @ pt
        pts_2d_a_aff.append(pt)
    pts_2d_a_aff = np.array(pts_2d_a_aff)

    pts_2d_b_aff = []
    for pt in pts_2d_b:
        pt = A_b @ pt
        pts_2d_b_aff.append(pt)
    pts_2d_b_aff = np.array(pts_2d_b_aff)

    # compute F and plot lines as before
    F_hat = F_least_squares(pts_2d_a_aff[:, :2], pts_2d_b_aff[:, :2])

   # denormalize F
    F = A_b.T @ F_hat @ A_a
    
    epilines_a = pts_2d_b @ F
    epilines_b = pts_2d_a @ F.T

    l_l = np.cross([0, 0, 1], [pic_a.shape[0], 0, 1])
    l_r = np.cross([0, pic_a.shape[1], 1], [pic_a.shape[0], pic_a.shape[1], 1])

    for l_a, l_b in zip(epilines_a, epilines_b):
        p_a_l = np.cross(l_a, l_l)
        p_a_r = np.cross(l_a, l_r)
        p_a_l = (p_a_l / p_a_l[-1]).astype(int)
        p_a_r = (p_a_r / p_a_r[-1]).astype(int)
        cv.line(pic_a, (p_a_l[:2]), (p_a_r[:2]), (0, 0, 255), thickness=2)
        p_b_l = np.cross(l_b, l_l)
        p_b_r = np.cross(l_b, l_r)
        p_b_l = (p_b_l / p_b_l[-1]).astype(int)
        p_b_r = (p_b_r / p_b_r[-1]).astype(int)
        cv.line(pic_b, (p_b_l[:2]), (p_b_r[:2]), (0, 0, 255), thickness=2)

    cv.imshow('', np.hstack((pic_b,pic_a))); cv.waitKey(0); cv.destroyAllWindows()

    # cv.imwrite("problem-sets/ps3/output/ps3-2-e-1.png", pic_a)
    # cv.imwrite("problem-sets/ps3/output/ps3-2-e-2.png", pic_b)

