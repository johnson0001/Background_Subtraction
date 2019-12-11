import cv2
import numpy as np
import os
import module1 as md1
import folder_pro as folder


np.seterr(divide='ignore', invalid='ignore')

def main():
    input_dir = "C:\\kanamitsu\\01" #処理するファイルのディレクトリ
    output_dir = "C:\\kanamitsu\\01_ver3.1" #出力するファイルのディレクトリ
    path_list, name_list, ext_list, out_list = folder.file_search(input_dir, output_dir)
    #back = cv2.imread(input_dir + "\\back2.bmp")
    back = cv2.imread("C:\\kanamitsu\\back.bmp")
    for file, outfile in zip(path_list, out_list):
        front = cv2.imread(file)
        mask = md1.Background_Subtraction(front, back)
        mask = md1.shadow_extract(back, front, mask)
        mask = md1.opening(mask, 3, 1)
        #mask = md1.closing(mask, 3, 1)
        cv2.imwrite(outfile, mask)


if __name__ == "__main__":
    main()


    ##影除去エラー対応
    #back_color = cv2.imread("C:\\sample_data\\back.bmp")
    #front_color = cv2.imread("C:\\sample_data\\MAH00008_mod_07091.bmp")
    #print(front_color.shape)
    #print(back_color.shape)
    #binary = cv2.imread("C:\\sample_bgsub_ver4_15\\MAH00008_mod_07091_out.bmp")
    #print(binary.shape)
    #md1.shadow_extract(back_color, front_color, binary)
    

