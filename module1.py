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

#ガウシアンフィルタ関数
def gaussian_filter(img, kernel):
    mask = cv2.filter2D(img, -1, kernel)
    return mask




#モルフォロジー処理（オープニング(収縮の後に膨張））
def opening(binary, ksize, n):
    kernel = np.ones((ksize, ksize), np.uint8)
    if n ==1:
        out = cv2.erode(binary, kernel, iterations=1)
        out = cv2.dilate(out, kernel, iterations=1)
        return out

    else:
        for i in range(n-1):
            if i == 0:
                out = cv2.erode(binary, kernel, iterations=1)
            else:
                out = cv2.erode(out, kernel, iterations=1)
        for i in range(n-1):
            out = cv2.dilate(out, kernel, iterations=1)
            
        return out

#モルフォロジー処理（クロージング（膨張の後に収縮））
def closing(binary, ksize, n):
    kernel = np.ones((ksize, ksize), np.uint8)
    if n == 1:
        out = cv2.dilate(binary, kernel, iterations=1)
        out = cv2.erode(out, kernel, iterations=1)
        return out

    else:
        for i in range(n-1):
            if i == 0:
                out = cv2.dilate(binary, kernel, iterations=1)
            else:
                out = cv2.dilate(out, kernel, iterations=1)
        for i in range(n-1):
            out = cv2.erode(out, kernel, iterations=1)

        return out





#影除去
def shadow_extract(back_colorimg, input_colorimg, binary_img):
    B1 = back_colorimg[:, :, 0:1].astype(np.double)
    G1 = back_colorimg[:, :, 1:2].astype(np.double)
    R1 = back_colorimg[:, :, 2:3].astype(np.double)

    B2 = input_colorimg[:, :, 0:1].astype(np.double)
    G2 = input_colorimg[:, :, 1:2].astype(np.double)
    R2 = input_colorimg[:, :, 2:3].astype(np.double)

    bunbo = np.sqrt(np.square(B1) + np.square(G1) + np.square(R1)) * np.sqrt(np.square(B2) + np.square(G2) + np.square(R2))
    bunsi = B1*B2 + G1*G2 + R1*R2
    cosine = bunsi / bunbo
    theta = np.arccos(cosine)
    threthold = 0.05

    shadow = np.where((theta < threthold) & (B1 > B2) & (G1 > G2) & (G1 > G2), 0, 255).astype(np.uint8)
    shadow = np.reshape(shadow, (shadow.shape[0], shadow.shape[1]))

    #binary_img = np.delete(binary_img, [1, 2], 2)
    #binary_img = np.reshape(binary_img, (binary_img.shape[0], binary_img.shape[1]))
    binary_out = np.where(shadow < 127, shadow, binary_img)
    #cv2.imshow("out", binary_out)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return binary_out





#背景差分関数
def Background_Subtraction(front_img, background_img):
    th = 15
    kernel = gaussian_kernel(5)

    mask = cv2.absdiff(front_img, background_img)#要素ごとの差の絶対値
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)#グレースケール化
    mask = gaussian_filter(mask, kernel)#ガウシアン化
    mask[mask < th] = 0
    mask[mask >= th] = 255

    #print(mask.shape)
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return mask