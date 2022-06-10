

from __future__ import division

import cv2
import numpy as np

from Tools.FakePad import fakePad

def illuminationCorrection(Image, kernel_size, Mask):
    #input: original RGB image and kernel size
    #output: illumination corrected RGB image
    ## The return can be a RGB 3 channel image, but better to show the user the green channel only
    ##since green channel is more clear and has higher contrast

    Mask = np.uint8(Mask)
    Mask[Mask > 0] = 1

    Img_pad = fakePad(Image, Mask, iterations=25)


    BackgroundIllumImage = cv2.medianBlur(Img_pad, ksize = kernel_size)

    maximumVal = np.max(BackgroundIllumImage)
    minimumVal = np.min(BackgroundIllumImage)
    constVal = maximumVal - 128

    BackgroundIllumImage[BackgroundIllumImage <=10] = 100
    IllumImage = Img_pad * (maximumVal / BackgroundIllumImage) - constVal
    IllumImage[IllumImage>255] = 255
    IllumImage[IllumImage<0] = 0
    IllumImage = np.uint8(IllumImage)

    IllumImage = cv2.medianBlur(IllumImage, ksize=3)

    return IllumImage


######################################################
def illuminationCorrection2(Image, kernel_size):
    #input: original RGB image and kernel size
    #output: illumination corrected RGB image
    ## The return can be a RGB 3 channel image, but better to show the user the green channel only
    ##since green channel is more clear and has higher contrast


    BackgroundIllumImage = cv2.medianBlur(Image, ksize = kernel_size)

    maximumVal = np.max(BackgroundIllumImage)
    minimumVal = np.min(BackgroundIllumImage)
    constVal = maximumVal - 128

    BackgroundIllumImage[BackgroundIllumImage <=10] = 100
    IllumImage = Image * (maximumVal / BackgroundIllumImage) - constVal
    IllumImage[IllumImage>255] = 255
    IllumImage[IllumImage<0] = 0
    IllumImage = np.uint8(IllumImage)

    IllumImage = cv2.medianBlur(IllumImage, ksize=3)

    return IllumImage
