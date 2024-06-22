import numpy as np

def normalize_01(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def normalize_11(img):
    return ((2 * (img - np.min(img))) / (np.max(img) - np.min(img))) - 1

def normalize_0255(img):
    return (255.0 * (img - np.min(img))) / (np.max(img) - np.min(img))

def gaussian_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)