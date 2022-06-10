

from __future__ import division
import cv2
import numpy as np
from skimage import measure
from Tools.BinaryPostProcessing import binaryPostProcessing3
from Tools.Float2Uint import float2Uint

"""This function uses hysteresis thresholding to segment the vessels,
resultImg is segmented in two levels.
it is better than adaptive thresholding, in that the vessel connectivity is better"""

def hysteresisThresholding(FilteredImg, Mask):
    """
    This function uses hysteresis thresholding to segment the vessels,
    resultImg is segmented in two levels.
    :param FilteredImg:
    :param Mask:
    :return:
    """

    FilteredImg = float2Uint(FilteredImg)

    ImgBW0 = cv2.adaptiveThreshold(FilteredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 35, -5)
    ImgBW0 = binaryPostProcessing3(ImgBW0, removeArea=100, fillArea=50)
    standardRatio = np.count_nonzero(ImgBW0) / float(np.count_nonzero(Mask))
    ##standardRatio is the vessel ratio of the adaptive thresholding method.


    upperRatio = standardRatio - 0.03
    lowerRatio = standardRatio + 0.01
    hist, bins = np.histogram(FilteredImg.ravel(), 100)
    upperthreshold = histogramThresholdMethod(hist, bins, upperRatio, Mask) #0.08
    lowerthreshold = histogramThresholdMethod(hist, bins, lowerRatio, Mask) #0.12
    AboveLower = FilteredImg >= lowerthreshold
    AboveUpper = FilteredImg >= upperthreshold

    AboveLower_Label = measure.label(AboveLower)
    Img_BW_hys = AboveLower.copy()
    for i, region in enumerate(measure.regionprops(AboveLower_Label)):
        if np.count_nonzero(AboveUpper[AboveLower_Label==i]) >= 100:
            pass
        else:
            Img_BW_hys[AboveLower_Label==i] = 0

    Img_BW_hys = np.uint8(Img_BW_hys)
    return Img_BW_hys




def histogramThresholdMethod(hist, bins, tempRatio, Mask):
    for i in range(len(hist), 0, -1):
        ratio = np.sum(hist[i:]) / np.count_nonzero(Mask)
        if ratio >= tempRatio:
            threshold = bins[i]
            break
    return threshold