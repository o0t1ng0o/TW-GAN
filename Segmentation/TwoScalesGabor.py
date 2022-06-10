
from __future__ import division

import cv2
import numpy as np

from Tools.BinaryPostProcessing import binaryPostProcessing3
from .GaborFiltering import gaborFiltering
from Tools.Im2Double import im2double

# import numba
# @numba.jit
def twoScalesGabor(Img_reverse, Mask):
    """
    Input: green channel Illumination corrected reverse image, Mask
    Output: Binary Image
    :param Img_reverse:
    :param Mask:
    :return:
    """

    Img_small = cv2.resize(Img_reverse, fx=0.35, fy=0.35, dsize=None)#dsize=(120, 120)
    Mask_small = cv2.resize(Mask, fx=0.35, fy=0.35, dsize=None)#dsize=(120, 120)

    height, width = Img_reverse.shape
    height_small, width_small = Img_small.shape

    Img_reverse_large = im2double(Img_reverse)
    Img_reverse_small = cv2.resize(Img_reverse_large, dsize=( width_small,height_small))

#####################################
    # pool = multiprocessing.Pool(processes=4)
    # pool_parameters = []
    # pool_parameters.append((Img_reverse_large, Mask))
    # pool_parameters.append((Img_reverse_small, Mask_small))
    #
    # gaborOutput = pool.map_async(gaborFiltering, pool_parameters)
    # gaborOutput = gaborOutput.get()
    #
    # filteredImg1_large, filteredImg2_large = gaborOutput[0]
    # filteredImg1_small, filteredImg2_small = gaborOutput[1]


    filteredImg1_large, filteredImg2_large = gaborFiltering((Img_reverse_large, Mask))
    filteredImg1_small, filteredImg2_small = gaborFiltering((Img_reverse_small, Mask_small))


    filteredImg2_smalltoLarge = cv2.resize(filteredImg2_small, dsize=(width, height))
    filteredImg2_final = np.maximum(filteredImg2_large, filteredImg2_smalltoLarge)
    filteredImg2_final_gaussian = cv2.GaussianBlur(filteredImg2_final, (5,5), 1)

    Img_BW0_final = cv2.adaptiveThreshold(filteredImg2_final_gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                  cv2.THRESH_BINARY, 35, -5)

    # Img_BW0_final = hysteresisThresholding(filteredImg2_final_gaussian, Mask)

    Img_BW_final = binaryPostProcessing3(Img_BW0_final, 300, 100)



    return Img_BW_final, filteredImg2_final_gaussian







