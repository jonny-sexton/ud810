import cv2 as cv
import numpy as np

def main():
    left = cv.imread("./stereo-corridor_l.png", cv.IMREAD_GRAYSCALE)
    right = cv.imread("./stereo-corridor_r.png", cv.IMREAD_GRAYSCALE)

    # cv.imshow("", left)
    # cv.waitKey(0)
    
    # normalize arrays
    left_norm = left / 255.0
    right_norm = right / 255.0

    # define image patch location
    patch_loc = [20, 70]
    patch_size = [50, 50]

    # extract patch from left image
    patch_left = left_norm[patch_loc[0]: patch_loc[0] + patch_size[0], patch_loc[1]: patch_loc[1] + patch_size[1]]

    # cv.imshow("", patch_left)
    # cv.waitKey(0)

    # extract strip from right image
    strip_right = right_norm[patch_loc[0]:(patch_loc[0] + patch_size[0]), :]

    # cv.imshow("", strip_right)
    # cv.waitKey(0)

    best_match = find_best_match(patch_left, strip_right)

    print("Best match found at pos:", best_match)

    best_patch = right_norm[patch_loc[0]: patch_loc[0] + patch_size[0], best_match:(best_match + patch_size[0])]

    # cv.imshow("", best_patch)
    # cv.waitKey(0)

    # define strip row (y) and square block size (b)
    y = 100
    b = 50

    # extract strips
    strip_left = left[y:(y + b), :]
    strip_right = right[y:(y + b), :]

    cv.imshow("", strip_left)
    cv.waitKey(0)

    cv.imshow("", strip_right)
    cv.waitKey(0)

    disparities = compute_disp(strip_left, strip_right, b)

    for block in disparities:
        print(block[0], block[1])

def find_best_match(patch, strip):

    # print(strip.shape)
    # print(patch.shape)
    best_fit = -1

    # get patch from place x on strip
    for i in range(strip.shape[1]- patch.shape[1]):
        temp_patch = strip[0 : patch.shape[0], i: patch.shape[1] + i]   

        # compare patch and temp patch, calculate ssd or something
        sum_abs = np.sum(np.abs(patch - temp_patch))

        if i == 0:
            sum_abs_prev = sum_abs

        elif sum_abs < sum_abs_prev:
            sum_abs_prev = sum_abs
            best_fit = i

        # print(i, sum_abs, sum_abs_prev)

    # return best fit
    return best_fit

def compute_disp(strip_left, strip_right, b):
    disparities = []

    # take first block from strip left and find it in strip right
    for i in range(strip_left.shape[1] // b):
        patch_left = strip_left[:, (i*b) : (i*b) + b]
        
        # cv.imshow("", patch_left)
        # cv.waitKey(0)

        best_match = find_best_match(patch_left, strip_right)

        # compute disparity at found block
        disparities.append([i*50, best_match])

    return disparities
    


if __name__ == "__main__":
    main()