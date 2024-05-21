import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, sobel, generic_gradient_magnitude
from scipy import datasets
from scipy.ndimage import shift

def noise(img, sigma):
    noise = np.random.randn(img.shape[0], img.shape[1]) * sigma
    img_noise = img + noise
    print(img_noise)
    return img_noise

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def onedim_filter(s, f):
    filterPos = -1
    for i in range(len(s)):
        matchFlag = True
        pos = i
        for j in range(len(f)):
            if s[i + j] == f[j]:
                continue
            else:
                matchFlag = False
                break
        if matchFlag == True:
            filterPos = pos
            break

    return filterPos

def twodim_filter(img, filter):
    print(img.shape)
    print(filter.shape)
    filterPos = -1
    # iter img pix
    for i in range(img.shape[0] - filter.shape[0]):
        for j in range(img.shape[1] - filter.shape[1]):
            # print("checking:", i, j)
            # check filter against image 
            matchFlag = True
            pos = (i, j)

            # check column
            for x in range(filter.shape[0]):
                # check row
                for y in range(filter.shape[1]):
                    if img[i + x, j + y] == filter[x, y]:
                        continue
                    else:
                        matchFlag = False
                        break
                if matchFlag == False:
                    break
            
            if matchFlag == True:
                filterPos = pos
                return filterPos
    
    return filterPos

def main():
    img = cv.imread("problem-sets/ps0/dolphin.jpg")
    img = cv.cvtColor(img, cv.IMREAD_GRAYSCALE).astype("float64")

    # img = datasets.ascent().astype('int32')

    glyph = img[234:270, 56:76]

    img_noise = noise(img, 12)
    img_noise = sp_noise(img, 0.05)
    img_noise = img_noise.astype("uint8")

    # apply filter
    img = gaussian_filter(img, sigma=1, mode='wrap')
    img_filtered = median_filter(img_noise, size=(3,3))
    img_filtered_x = sobel(img, 0).astype("float64")
    img_filtered_y = sobel(img, 1).astype("float64")

    # cv.imshow("",img_filtered)
    # # Waits for a keystroke
    # cv.waitKey(0)

    # s = [-1, 0, 0, 1, 1, 1, 0, -1, -1, 0, 1, 0, 0, -1]
    # t = [1, 0, -1]
    
    # print(onedim_filter(s, t))

    # xy derivatives
    # magnitude = np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    # magnitude *= 255.0 / np.max(magnitude)  # normalization
    # magnitude = magnitude.astype("uint8")

    direction = np.arctan2(img_filtered_y, img_filtered_x)
    direction += np.abs(np.min(direction)) # shift values
    direction *= 255.0 / np.max(direction) # normalize [0,1]
    direction = direction.astype("uint8")

    img_filtered_x += np.abs(np.min(img_filtered_x))
    img_filtered_y += np.abs(np.min(img_filtered_y))

    # img_filtered_x *= 255.0 / np.max(img_filtered_x)
    # img_filtered_y *= 255.0 / np.max(img_filtered_y)

    cv.imshow("",img_filtered_x)
    cv.waitKey(0)

    # frizzy = cv.imread("frizzy.png")
    # froomer = cv.imread("froomer.png")

    # frizzy = cv.cvtColor(frizzy, cv.COLOR_RGB2GRAY).astype("int32")
    # froomer = cv.cvtColor(froomer, cv.COLOR_RGB2GRAY).astype("int32")
    
    # frizzy_smooth = gaussian_filter(frizzy, sigma=1, mode='wrap')
    # froomer_smooth = gaussian_filter(froomer, sigma=1, mode='wrap')

    # frizzy_smooth_x = sobel(frizzy_smooth, 0).astype("float64")
    # frizzy_smooth_y = sobel(frizzy_smooth, 1).astype("float64")
    # frizzy_smooth_xy = frizzy_smooth_x + frizzy_smooth_y
    # froomer_smooth_x = sobel(froomer_smooth, 0).astype("float64")
    # froomer_smooth_y = sobel(froomer_smooth, 1).astype("float64")
    # froomer_smooth_xy = froomer_smooth_x + froomer_smooth_y

    #threshold arrays
    # frizzy_smooth_x = (frizzy_smooth_x > 128) * frizzy_smooth_x
    # frizzy_smooth_y = (frizzy_smooth_y > 128) * frizzy_smooth_y
    # frizzy_smooth_xy = (frizzy_smooth_xy > 128) * frizzy_smooth_xy
    # froomer_smooth_x = (froomer_smooth_x > 128) * froomer_smooth_x
    # froomer_smooth_y = (froomer_smooth_y > 128) * froomer_smooth_y
    # froomer_smooth_xy = (froomer_smooth_xy > 128) * froomer_smooth_xy

    # xy derivatives
    # frizzy_magnitude = np.sqrt(frizzy_smooth_x**2 + frizzy_smooth_y**2)
    # frizzy_magnitude *= 255.0 / np.max(frizzy_magnitude)  # normalization
    # frizzy_magnitude = frizzy_magnitude.astype("uint8")
    # froomer_magnitude = np.sqrt(froomer_smooth_x**2 + froomer_smooth_y**2)
    # froomer_magnitude *= 255.0 / np.max(froomer_magnitude)  # normalization
    # froomer_magnitude = froomer_magnitude.astype("uint8")

    # frizzy_direction = np.arctan2(frizzy_smooth_y, frizzy_smooth_x)
    # frizzy_direction += np.abs(np.min(frizzy_direction)) # shift values
    # frizzy_direction *= 255.0 / np.max(frizzy_direction) # normalize [0,1]
    # frizzy_direction = frizzy_direction.astype("uint8")
    # froomer_direction = np.arctan2(froomer_smooth_y, froomer_smooth_x)
    # froomer_direction += np.abs(np.min(froomer_direction)) # shift values
    # froomer_direction *= 255.0 / np.max(froomer_direction) # normalize [0,1]
    # froomer_direction = froomer_direction.astype("uint8")

    # frizzy_smooth_x *= 255.0 / np.max(frizzy_smooth_x)
    # frizzy_smooth_y *= 255.0 / np.max(frizzy_smooth_y)
    # froomer_smooth_x *= 255.0 / np.max(froomer_smooth_x)
    # froomer_smooth_y *= 255.0 / np.max(froomer_smooth_y)

    # common = np.logical_and(frizzy_magnitude, froomer_magnitude).astype("uint8")
    # common *= 255
    # common = common.astype("uint8")
    # print(froomer_smooth_xy[150:200, 150:200])

    # print(common[150:200, 150:200])


    # cv.imshow("",froomer_smooth_x)
    # cv.waitKey(0)
    # cv.imshow("",froomer_smooth_y)
    # cv.waitKey(0)
    # cv.imshow("",froomer_magnitude)
    # cv.waitKey(0)
    # cv.imshow("",frizzy_magnitude)
    # cv.waitKey(0)
    # cv.imshow("",common)
    # cv.waitKey(0)

    # print(twodim_filter(img, glyph))





if __name__ == "__main__":
    main()