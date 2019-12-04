import cv2
import numpy as np

#平均0, 分散 sigma^2の二次元ガウス分布
def norm2d(x, y, sigma):
    Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return Z


#ガウシアンカーネル
def gaussian_kernel(size):
    if size%2==0:
        print('kernel size should be odd')
        return

    sigma = (size-1)/2

    #[0,size]→[-sigma, sigma]にずらす
    x = y = np.arange(0,size) - sigma
    X, Y = np.meshgrid(x, y)

    mat = norm2d(X, Y, sigma)

    #総和が1になるように
    kernel = mat / np.sum(mat)
    return kernel


#ガウシアンフィルタ
def gaussian_filter(img, kernel):
    mask = cv2.filter2D(img, -1, kernel)
    return mask


#背景差分関数
def Background_Subtraction(front_image, background_image):
    th = 40
    mask = cv2.absdiff(front_image, background_image)#要素ごとの差の絶対値
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)#グレースケール化
    mask = gaussian_filter(mask, gaussian_kernel(5))#ガウシアン化
    mask[mask < th] = 0
    mask[mask >= th] = 255

    return mask