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

#モルフォロジー処理（テスト用）
def morphology_test(binary, ksize):
    kernel = np.ones((ksize, ksize), np.uint8)
    out = cv2.erode(binary, kernel, iterations=1)
    out = cv2.dilate(out, kernel, iterations=1)
    out = cv2.dilate(out, kernel, iterations=1)
    
    return out




#影除去
def shadow_extract(back_colorimg, input_colorimg, binary):
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

    binary_out = np.where(shadow < 127, shadow, binary)

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

    return mask





#人物領域を任意の範囲で抽出
def triming(img):
    white = np.argwhere(img!=0) #画素値が0でない座標の配列

    if(white.shape != (0, 2)): #白領域が存在する場合のみ処理
        y_white = white[:,0] #画素値が白のピクセルのy座標一覧
        x_white = white[:,1] #画素値が白のピクセルのx座標一覧

        horizontal_center = np.mean(x_white) #水平方向の中心
        vertical_center = np.mean(y_white) #垂直方向の中心
        horizontal_center = np.round(horizontal_center).astype(np.int) #実数→整数
        vertical_center = np.round(vertical_center).astype(np.int)
        y_min = np.min(y_white) #白領域のy座標が最小のピクセルのy座標
        y_max = np.max(y_white) #白領域のy座標が最大のピクセルのy座標
        #トリミング
        img = img[vertical_center-210:vertical_center+180, horizontal_center-100:horizontal_center+130] #任意の範囲を指定
        return img


#人物領域を外接矩形で抽出
def contours(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]     
    cnt = max(contours, key=lambda x:cv2.contourArea(x))
    x, y, width, height = cv2.boundingRect(cnt)
    img = img[y:y+height, x:x+width]
    return img
