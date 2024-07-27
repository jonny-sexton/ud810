from ps7_functions import *
import cv2 as cv
import numpy as np
import time

from collections import OrderedDict
import sys
sys.path.append('./solutions/utils/')
from toolbox import *

from ps7_functions import *

def ps7_1_a():
    vid = "solutions/ps7/input/PS7A1P1T1.avi"

    # parse video
    cap = cv.VideoCapture(vid)

    # get window of face from first frame of video
    ret, frame = cap.read()

    # define params
    theta = 0.1
    tau = 10

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    model = normalize_01(gray)
    gray_normalized = model.copy()
    gray_normalized = cv.GaussianBlur(gray_normalized,(9,9), 0)

    frame_count = 0

    while cap.isOpened():

        gray_normalized_t_0 = gray_normalized

        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_normalized = normalize_01(gray)
        gray_normalized = cv.GaussianBlur(gray_normalized,(9,9), 0)

        for i in range(model.shape[0]):
            for j in range(model.shape[1]):
                diff_px = gray_normalized_t_0[i][j] - gray_normalized[i][j]
                if np.abs(diff_px) > theta:
                    model[i][j] = tau
                else:
                    model[i][j] = max(model[i][j] - 1,0)

        # cv.imshow("", normalize_0255(model).astype("uint8")); cv.waitKey(0)

        # save frame
        frames = [10,20,30]
        print(frame_count)
        if frame_count in frames:
            fp = "solutions/ps7/output/1-a-" + str(frame_count) + ".png"
            cv.imwrite(fp, normalize_0255(model).astype("uint8"))

        frame_count +=1 

def ps7_1_b():
    mhi = MHI("solutions/ps7/input/PS7A1P1T1.avi", "1-b-1-", [80], 0.1, 100)
    mhi.compute_mhi()

    mhi = MHI("solutions/ps7/input/PS7A2P1T1.avi", "1-b-2-", [35], 0.1, 20)
    mhi.compute_mhi()

    mhi = MHI("solutions/ps7/input/PS7A3P1T1.avi", "1-b-3-", [40], 0.1, 50)
    mhi.compute_mhi()

def ps7_2_a():
    a = MHI("solutions/ps7/input/PS7A1P2T1.avi", "2-a-a-", [10], 0.1, 10)
    a.compute_mhi()
    D_test_a = a.compute_moments("nu")

    b = MHI("solutions/ps7/input/PS7A2P2T1.avi", "2-a-b-", [35], 0.1, 20)
    b.compute_mhi()
    D_test_b = b.compute_moments("nu")

    c = MHI("solutions/ps7/input/PS7A3P2T1.avi", "2-a-c-", [35], 0.1, 20)
    c.compute_mhi()
    D_test_c = c.compute_moments("nu")

    i = MHI("solutions/ps7/input/PS7A1P1T1.avi", "2-a-i-", [20], 0.1, 10)
    i.compute_mhi()
    D_test_i = i.compute_moments("nu")

    j = MHI("solutions/ps7/input/PS7A2P1T1.avi", "2-a-j-", [35], 0.1, 20)
    j.compute_mhi()
    D_test_j = j.compute_moments("nu")

    k = MHI("solutions/ps7/input/PS7A3P1T1.avi", "2-a-k-", [40], 0.1, 20)
    k.compute_mhi()
    D_test_k = k.compute_moments("nu")

    test = np.sum((D_test_a-D_test_k)**2)

    print(test)

    C = np.array([[np.sum((D_test_a-D_test_k)**2), np.sum((D_test_b-D_test_k)**2), np.sum((D_test_c-D_test_k)**2)],
                  [np.sum((D_test_a-D_test_j)**2), np.sum((D_test_b-D_test_j)**2), np.sum((D_test_c-D_test_j)**2)],
                  [np.sum((D_test_a-D_test_i)**2), np.sum((D_test_b-D_test_i)**2), np.sum((D_test_c-D_test_i)**2)]])
    
    print(np.around(C, decimals=2))

    # # compute A1 confusion matrix
    # mhi = MHI("solutions/ps7/input/PS7A1P1T1.avi", "", [80], 0.1, 100)
    # mhi.compute_mhi()
    # D_A1P1T1 = mhi.compute_moments("nu")
    # D_A1P1T1_i = np.sum((D_A1P1T1-D_test_i)**2)
    # # D_A1P1T1_j = np.sum((D_A1P1T1-D_test_j)**2)
    # # D_A1P1T1_k = np.sum((D_A1P1T1-D_test_k)**2)

    
    # C = np.array([[D_A1P1T1_i, D_A1P1T2_i, D_A1P1T3_i], [D_A1P2T1_i, D_A1P2T2_i, D_A1P2T3_i], [D_A1P3T1_i, D_A1P3T2_i, D_A1P3T3_i]])

    # print(C)

    """
    TEXT RESPONSE:
    a, b, c = PS7A1P2T1, PS7A2P2T1, PS7A3P2T1
    i, j, k = PS7A1P1T1, PS7A2P1T1, PS7A3P1T1
    [[(a,k), (b,k), (c,k)],
    [(a,j),(b,j),(c,j)],
    [(a,i),(b,i),(c,i)]]
    [[ 474.85    6.82 1880.62]
    [ 398.53   13.41 1781.97]
    [ 183.     95.73 1533.1 ]]
    """





def ps7_2_b():
    pass

def ps7_3_a():
    pass

ps7_list = OrderedDict([('1a', ps7_1_a), ('1b', ps7_1_b), ('2a', ps7_2_a), ('2b', ps7_2_b), ('3a', ps7_3_a)])

if __name__=="__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] in ps7_list:
            print('\nExecuting task %s\n=================='%sys.argv[1])
            ps7_list[sys.argv[1]]()
        else:
            print('\nGive argument from list {1a,1b,2a,2b,3a} for the corresponding task')
    else:
        print('\n * Executing all tasks: * \n')
        for idx in ps7_list.keys():
            print('\nExecuting task: %s\n=================='%idx)
            ps7_list[idx]()