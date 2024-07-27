import cv2 as cv
import numpy as np

import sys
sys.path.append('./solutions/utils/')
from toolbox import *

class MHI():
    def __init__(self, video, q_n="", frames=[], theta=0.1, tau=15):
        self.video = video
        self.q_n = q_n
        self.frames = frames
        self.theta = theta
        self.tau = tau
        self.nus_pq = []
        self.mus_pq = []
        self.nus_pq_thresh = []
        self.mus_pq_thresh = []

    def compute_mhi(self):
        # parse video
        cap = cv.VideoCapture(self.video)

        # get window of face from first frame of video
        ret, frame = cap.read()

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
                    if np.abs(diff_px) > self.theta:
                        model[i][j] = self.tau
                    else:
                        model[i][j] = max(model[i][j] - 1,0)

            # save frame
            if frame_count in self.frames:
                if self.q_n != "":
                    fp = "solutions/ps7/output/" + self.q_n + str(frame_count) + ".png"
                    cv.imwrite(fp, normalize_0255(model).astype("uint8"))

                # compute threshold
                ret,model_thresh = cv.threshold(model,0.1,1.0,cv.THRESH_BINARY)

                # compute M
                model = normalize_01(model) * 0.1 # normalize values
                M_00 = self.compute_M(model, 0, 0)
                M_10 = self.compute_M(model, 1, 0)
                M_01 = self.compute_M(model, 0, 1)

                # compute M thresh
                M_00_thresh = self.compute_M(model_thresh, 0, 0)
                M_10_thresh = self.compute_M(model_thresh, 1, 0)
                M_01_thresh = self.compute_M(model_thresh, 0, 1)

                x_ = M_10 / M_00
                y_ = M_01 / M_00

                x_thresh = M_10_thresh / M_00_thresh
                y_thresh = M_01_thresh / M_00_thresh

                # print(x_,y_)

                mus = [(2,0), (0,2), (1,2), (2,1), (2,2), (3,0), (0,3)]

                mu_00 = self.compute_mu(model, x_, y_, 0, 0)
                mu_00_thresh = self.compute_mu(model_thresh, x_thresh, y_thresh, 0, 0)

                for mu in mus:
                    mu_pq = self.compute_mu(model, x_, y_, mu[0], mu[1])
                    mu_pq_thresh = self.compute_mu(model_thresh, x_thresh, y_thresh, mu[0], mu[1])
                    self.mus_pq.append(mu_pq)
                    self.mus_pq_thresh.append(mu_pq_thresh)

                for i in range(len(self.mus_pq)):
                    mu_pq = self.mus_pq[i]
                    mu_pq_thresh = self.mus_pq_thresh[i]
                    pq = mus[i]

                    nu_pq = self.compute_nu(mu_00, mu_pq, pq[0], pq[1])
                    nu_pq_thresh = self.compute_nu(mu_00_thresh, mu_pq_thresh, pq[0], pq[1])

                    self.nus_pq.append(nu_pq)
                    self.nus_pq_thresh.append(nu_pq_thresh)

            frame_count +=1 

    def compute_M(self, model, i, j):
        M = np.zeros(model.shape)

        for y in range(model.shape[0]):
            for x in range(model.shape[1]):
                if model[y][x] > 0:
                    M[y][x] = ((x**i) * (y**j)) * model[y][x]
        
        return np.sum(M)
    
    def compute_mu(self, model, x_, y_, p, q):
        mu = np.zeros(model.shape)

        for y in range(model.shape[0]):
            for x in range(model.shape[1]):
                if model[y][x] > 0:
                    mu[y][x] = (((x-x_)**p) * ((y-y_)**q)) * model[y][x]

        return np.sum(mu)

    def compute_nu(self, mu_00, mu_pq, p, q):
        return mu_pq / (mu_00**(1 + ((p+q)/2)))
    
    def compute_moments(self, mode=""):
        if self.nus_pq == []:
            self.compute_mhi()
        if mode == "nu":
            return np.array([self.nus_pq, self.nus_pq_thresh])
        elif mode == "mu":
            return np.array([self.mus_pq, self.mus_pq_thresh])
        else:
            print("Select either nu or mu mode!")