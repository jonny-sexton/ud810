import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

def proj_3d_2d(coord_3d, f):
    x = coord_3d[0]
    y = coord_3d[1]
    z = coord_3d[2]

    return ((f*x)/z, (f*y)/z)

def main():
    coord_3d = [200,100,120]
    f=50
    
    print(proj_3d_2d(coord_3d, 50))


if __name__ == "__main__":
    main()