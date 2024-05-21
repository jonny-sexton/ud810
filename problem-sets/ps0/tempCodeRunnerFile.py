img = cv.imread("dolphin.jpg")
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype("int32")

    img = datasets.ascent().astype('int32')

    glyph = img[234:270, 56:76]

    img_noise = noise(img, 12)
    img_noise = sp_noise(img, 0.05)
    img_noise = img_noise.astype("uint8")

    apply filter
    img = gaussian_filter(img, sigma=1, mode='wrap')
    img_filtered = median_filter(img_noise, size=(3,3))
    img_filtered_x = sobel(img, 0).astype("float64")
    img_filtered_y = sobel(img, 1).astype("float64")

    cv.imshow("",img_filtered)
    Waits for a keystroke
    cv.waitKey(0)
