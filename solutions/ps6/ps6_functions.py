import numpy as np
import cv2

class Video_Tracker_PF():
    def __init__(self, model, search_space, num_particles=100, state_dims=2,
                 control_std=10, sim_std=20, alpha=0.0):
        self.model = model
        self.search_space = search_space[::-1]
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.control_std = control_std
        self.sim_std = sim_std
        self.alpha = alpha
        # inialize particles using a uniform distribution
        self.particles = np.array([np.random.uniform(0, self.search_space[i],
                                                     self.num_particles)
                                   for i in range(self.state_dims)]).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)
        self.idxs = np.arange(num_particles)
        self.estimate_state()

    def estimate_state(self):
        state_idx = np.random.choice(self.idxs, 1, p=self.weights)
        self.state = self.particles[state_idx][0]

    def update(self, frame):
        self.displace()
        self.observe(frame)
        self.resample()
        self.estimate_state()
        if self.alpha > 0:
            self.update_model(frame)

    def displace(self):
        # displace particles using a normal distribution centered around 0
        self.particles += np.random.normal(0, self.control_std,
                                           self.particles.shape)
        # np.random.normal(particle_uv[0],10,1)[0])

    def observe(self, img):
        mh, mw = self.model.shape[:2]
        minx = (self.particles[:,0] - mw/2).astype(np.int32)
        miny = (self.particles[:,1] - mh/2).astype(np.int32)
        candidates = [img[miny[i]:miny[i]+mh, minx[i]:minx[i]+mw] # compute patches
                      for i in range(self.num_particles)]
        # compute importance weight - similarity of each patch to the model
        self.weights = np.array([self.similarity(cand, self.model, self.sim_std)
                                for cand in candidates])
        # normalize the weights
        self.weights /= np.sum(self.weights)
        pass

    def resample(self):
        sw, sh = self.search_space[:2]
        mh, mw = self.model.shape[:2]
        # sample new particle indices using the distribution of the weights
        j = np.random.choice(self.idxs, self.num_particles, True,
                             p=self.weights.T)
        # get a random control input from a normal distribution
        # control = np.random.normal(0, self.control_std, self.particles.shape)
        # sample the particles using the distribution of the weights
        self.particles = np.array(self.particles[j])
        # clip particles in case the window goes out of the image limits
        self.particles[:,0] = np.clip(self.particles[:,0], 0, sw - 1)
        self.particles[:,1] = np.clip(self.particles[:,1], 0, sh - 1)

    def similarity(self, img1, img2, std=10):
        if np.subtract(img1.shape, img2.shape).any():
            return 0
        else:
            mse = np.sum(np.subtract(img1, img2, dtype=np.float32) ** 2)
            mse /= float(img1.shape[0] * img1.shape[1])
            return np.exp(-mse / 2 / std**2)
        
    def visualize_filter(self, img):
        self.draw_particles(img)
        self.draw_window(img)
        # self.draw_std(img)

    def draw_particles(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p.astype(int)), 2, (180,255,0), -1)

    def draw_window(self, img):
        best_idx = cv2.minMaxLoc(self.weights)[3][1]
        best_state = self.particles[best_idx]
        pt1 = (best_state - np.array(self.model.shape[::-1])/2).astype(np.int32)
        #  pt1 = (self.state - np.array(self.model.shape[::-1])/2).astype(np.int)
        pt2 = pt1 + np.array(self.model.shape[::-1])
        cv2.rectangle(img, tuple(pt1), tuple(pt2), (0,255,0), 2)

    def update_model(self, frame):
        # get current model based on belief
        mh, mw = self.model.shape[:2]
        minx = int(self.state[0] - mw/2)
        miny = int(self.state[1] - mh/2)
        best_model = frame[miny:miny+mh, minx:minx+mw]
        # apply appearance model update if new model shape is unchanged
        if best_model.shape == self.model.shape:
            self.model = self.alpha * best_model + (1-self.alpha) * self.model
            self.model = self.model.astype(np.uint8)

class Video_Tracker_PFMSLPF:
    def __init__(self, model, search_space, num_particles=100, state_dims=2,
                 control_std=10, sim_std=20, alpha=0.0):
        self.model = model
        self.search_space = search_space[::-1]
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.control_std = control_std
        self.sim_std = sim_std
        self.alpha = alpha
        # inialize particles using a uniform distribution
        self.particles = np.array([np.random.uniform(0, self.search_space[i],
                                                     self.num_particles)
                                   for i in range(self.state_dims)]).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)
        self.idxs = np.arange(num_particles)
        self.estimate_state()

    def update(self, frame):
        self.displace()
        self.observe(frame)
        self.resample()
        self.estimate_state()
        if self.alpha > 0:
            self.update_model(frame)

    def displace(self):
        # displace particles using a normal distribution centered around 0
        self.particles += np.random.normal(0, self.control_std,
                                           self.particles.shape)

    def observe(self, img):
        # get patches corresponding to each particle
        mh, mw = self.model.shape[:2]
        minx = (self.particles[:,0] - mw/2).astype(np.int32)
        miny = (self.particles[:,1] - mh/2).astype(np.int32)
        candidates = [img[miny[i]:miny[i]+mh, minx[i]:minx[i]+mw]
                      for i in range(self.num_particles)]
        # compute importance weight - similarity of each patch to the model
        self.weights[:] = [compare_hist(cand, self.model, std=10,
                                              num_bins=8)
                                 for cand in candidates]
        # normalize the weights
        self.weights /= self.weights.sum()

    def resample(self):
        sw, sh = self.search_space[:2]
        mh, mw = self.model.shape[:2]
        # sample new particle indices using the distribution of the weights
        distrib = self.weights.flatten()
        j = np.random.choice(self.idxs, self.num_particles, True, p=distrib)
        # get a random control input from a normal distribution
        control = np.random.normal(0, self.control_std, self.particles.shape)
        # sample the particles using the distribution of the weights
        self.particles = np.array(self.particles[j])
        # clip particles in case the window goes out of the image limits
        self.particles[:,0] = np.clip(self.particles[:,0], 0, sw - 1)
        self.particles[:,1] = np.clip(self.particles[:,1], 0, sh - 1)

    def estimate_state(self):
        state_idx = np.random.choice(self.idxs, 1, p=self.weights.flatten())
        self.state = self.particles[state_idx][0]

    def update_model(self, frame):
        # get current model based on belief
        mh, mw = self.model.shape[:2]
        minx = int(self.state[0] - mw/2)
        miny = int(self.state[1] - mh/2)
        best_model = frame[miny:miny+mh, minx:minx+mw]
        # apply appearance model update if new model shape is unchanged
        if best_model.shape == self.model.shape:
            self.model = self.alpha * best_model + (1-self.alpha) * self.model
            self.model = self.model.astype(np.uint8)

    def visualize_filter(self, img):
        self.draw_particles(img)
        self.draw_window(img)
        self.draw_std(img)

    def draw_particles(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p.astype(int)), 2, (180,255,0), -1)

    def draw_window(self, img):
        best_idx = cv2.minMaxLoc(self.weights)[3][1]
        best_state = self.particles[best_idx]
        pt1 = (best_state - np.array(self.model.shape[1::-1])/2).astype(np.int32)
        #  pt1 = (self.state - np.array(self.model.shape[1::-1])/2).astype(np.int)
        pt2 = pt1 + np.array(self.model.shape[1::-1])
        cv2.rectangle(img, tuple(pt1), tuple(pt2), (0,255,0), 2)

    def draw_std(self, img):
        weighted_sum = 0
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.ravel())
        cv2.circle(img, tuple(self.state.astype(np.int32)),
                   int(weighted_sum), (255,255,255), 1)

def compare_hist(img1, img2, std=10, num_bins=8):
    if np.subtract(img1.shape, img2.shape).any():
        return 0.0
    else:
        x = chisqr(img1, img2, num_bins)
        return np.exp(-x / 2)


def chisqr(img1, img2, num_bins=8):
        hist1 = np.zeros(1*num_bins, dtype=np.float32)
        hist2 = np.zeros(1*num_bins, dtype=np.float32)
        K = num_bins
        for i in range(1):
            hist1[i*K:i*K+K] = cv2.calcHist(img1, [i], None, [num_bins], [0,256]).T
            hist2[i*K:i*K+K] = cv2.calcHist(img2, [i], None, [num_bins], [0,256]).T
            hist1[i*K:i*K+K] /= hist1[i*K:i*K+K].sum()
            hist2[i*K:i*K+K] /= hist2[i*K:i*K+K].sum()

        c = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        #  c = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10)
                          #  for (a, b) in zip(hist1, hist2)])
        return c

