import cv2
import numpy as np
import os
import module1 as md1
import folder_pro as folder


np.seterr(divide='ignore', invalid='ignore')

def main():
    input_dir = "C:\\onishi\\04\\04" #処理するファイルのディレクトリ
    output_dir = "C:\\onishi\\04\\04extra" #出力するファイルのディレクトリ
    path_list, name_list, ext_list, out_list = folder.file_search(input_dir, output_dir)
    back = cv2.imread("C:\\onishi\\back\\back3.bmp")
    for file, outfile in zip(path_list, out_list):
        front = cv2.imread(file)
        mask = md1.Background_Subtraction(front, back)
        mask = md1.shadow_extract(back, front, mask)
        mask = md1.opening(mask, 3, 3)
        mask = md1.closing(mask, 3, 3)
        cv2.imwrite(outfile, mask)


if __name__ == "__main__":
    #main()


    ##FOR_TEST
    #back_color = cv2.imread("C:\\hayakawa\\back.bmp")
    #print(back_color.shape)
    #front_color = cv2.imread("C:\\hayakawa\\01\\01\\MAH00009_mod_16400.bmp")
    #print(front_color.shape)
    #extra_img = md1.Background_Subtraction(front_color, back_color)
    #shadow_extra_img = md1.shadow_extract(back_color, front_color, extra_img)
    #cv2.imwrite("C:\\kanamitsu\\shadow_extra_img.bmp", shadow_extra_img)
    #binary = cv2.imread("C:\\kanamitsu\\shadow_extra_img.bmp", 0) #imreadはデフォルトでカラー画像（3次元配列）として読み込まれる
    #print(binary.shape)
    #print(binary[100][100])
    #kernel = np.ones((3, 3), np.uint8)
    #binary = cv2.erode(binary, kernel, iterations=1)
    #binary = cv2.dilate(binary, kernel, iterations=1)
    #binary = cv2.dilate(binary, kernel, iterations=1)
    #binary = cv2.erode(binary, kernel, iterations=1)
    #binary = cv2.erode(binary, kernel, iterations=1)
    #binary = cv2.dilate(binary, kernel, iterations=1)
    #cv2.imshow("out", shadow_extra_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img = cv2.imread("C:\\sample\\01extra\\MAH00009_mod_09345_out.bmp", 0)
    md1.scaling(img)