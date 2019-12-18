import cv2
import numpy as np
import os
import module1 as md1
import folder_pro as folder


np.seterr(divide='ignore', invalid='ignore')

#人物領域（白）と背景領域（黒）の二値化   
##入力：前景・背景のカラー画像     #出力：二値化画像
def Binarization():
    input_dir = "C:\\sample\\01extra" #処理するファイルのディレクトリ
    output_dir = "C:\\sample\\triming" #出力するファイルのディレクトリ
    path_list, name_list, ext_list, out_list = folder.file_search(input_dir, output_dir)
    back = cv2.imread("C:\\onishi\\back\\back3.bmp") #背景カラー画像
    for file, outfile in zip(path_list, out_list):
        front = cv2.imread(file)
        mask = md1.Background_Subtraction(front, back) #背景差分
        mask = md1.shadow_extract(back, front, mask) #影除去
        mask = md1.opening(mask, 3, 3) #モルフォロジー
        mask = md1.closing(mask, 3, 3) #モルフォロジー
        cv2.imwrite(outfile, mask)


#人物領域の抽出　#処理領域の削減
##入力：二値化画像      #出力：二値化画像
def Triming():
    input_dir = "C:\\sample\\01extra" #処理するファイルのディレクトリ
    output_dir = "C:\\sample\\triming2" #出力するファイルのディレクトリ
    path_list, name_list, ext_list, out_list = folder.file_search(input_dir, output_dir)
    for file, outfile in zip(path_list, out_list):
        img = cv2.imread(file, 0) #二値化画像（二次元配列）で読み込む
        #img = md1.contours(img) #外接矩形で人物領域抽出
        img = md1.triming(img) #任意の範囲で人物領域抽出
        cv2.imwrite(outfile, img)




if __name__ == "__main__":
    #Binarization()
    #Triming()


    ##FOR_TEST

    #cv2.imshow("out", shadow_extra_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #img = cv2.imread("C:\\sample\\01extra\\MAH00009_mod_09345_out.bmp", 0)
    #print(img.shape)
    #md1.triming(img)

    img = cv2.imread("C:\\sample\\01extra\\MAH00009_mod_09345_out.bmp", 0)
    md1.contours(img)