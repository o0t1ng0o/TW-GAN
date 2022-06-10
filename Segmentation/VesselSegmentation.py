
from __future__ import division

import cv2
import numpy as np
from skimage import morphology

from .TwoScalesGabor import twoScalesGabor
from Tools.FakePad import fakePad
from Tools.Im2Double import im2double


def vesselSegmentation(IllumImage, Mask, CR_flag = True):
    """
    Vessel segmentation module.
    :param IllumImage:
    :param Mask:
    :param CR_flag:
    :return:
    """


    ####Input:
    ####       RGB or green channel Illumination corrected Image,
    ####       Mask
    ####Output: The binary vessel image and skeleton Image

    if len(IllumImage.shape) == 3:  # if the image is RGB image, get the green channel only
        IllumImage_green = IllumImage[:, :, 1]
    else:
        IllumImage_green = IllumImage

    Mask = np.uint8(Mask)
    Mask[Mask>0] = 1

    if CR_flag == True: ##if CR_flag is true, fill the central reflex.
        IllumImage_green = morphology.opening(IllumImage_green, morphology.disk(3))
        IllumImage_green = cv2.medianBlur(IllumImage_green, ksize=3)
    else:
        pass

    # Img_green_pad = fakePad(IllumImage_green, Mask, iterations=35)

    Img_green_reverse0 = 255 - IllumImage_green
    Img_green_reverse0 = cv2.medianBlur(Img_green_reverse0, ksize=5)
    Img_green_reverse = im2double(Img_green_reverse0)


    ##Get the binary vessel map
    Img_BW, filteredImg2_final_gaussian = twoScalesGabor(Img_green_reverse, Mask)
    Img_BW = cv2.bitwise_and(Img_BW, Img_BW, mask=Mask)

    Img_BW[:20, :] = 0
    Img_BW[-20:, :] = 0
    Img_BW[:, :20] = 0
    Img_BW[:, -20:] = 0


    return Img_BW

