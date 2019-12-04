import cv2
import numpy as np
import os
import module1 as md1


def main():
    #os.chdir('C:\\sampledata_bgsub')
    img_inpath = "C:\\sampledata_bgsub\\sampledata1\\"
    img_outpath = "C:\\sampledata_bgsub\\sampledata1_pro\\"
    back = cv2.imread(img_inpath + "back.bmp")
    front = cv2.imread(img_inpath + "front1.bmp")

    #print(np.array(back))
    #mask = md1.gaussian_filter(front, md1.gaussian_kernel(5))


    mask = md1.Background_Subtraction(front, back)


    #cv2.imshow('sample', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(img_outpath + "mask3_40.bmp", mask)



if __name__ == "__main__":

    main()
 
    #print(md1.gaussian_kernel(5))