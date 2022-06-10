


from __future__ import division

import cv2
import numpy as np
from skimage import restoration

from Tools.Standardize import standardize
from Segmentation.GetLineMask import getLinemask
from scipy.ndimage.filters import convolve, correlate
from Tools.Im2Double import im2double
from Tools.Float2Uint import float2Uint
import numba

@numba.jit
def lineDetector(Image, Mask, threshold):
    """
    #Vessel Segmentation Using Alaudin's method: Line Detector

    :param Image:
    :param Mask:
    :param threshold:
    :return: Img_BW0, ResultImg
    """


    Img = im2double(Image)
    height, width = Img.shape

    winSize = 15
    step = 2

    features = standardize(Img, Mask, 0)

    filterNumber = 18
    for i in range(1, winSize+1, step):
        L = i
        # R = get_lineresponse(DilatedImg, winSize, L)
        avgresponse = cv2.blur(Img, ksize=(winSize, winSize))
        lineResponzes = np.ndarray((height, width, filterNumber))

        for m in range(0, filterNumber):
            theta = m * 180/filterNumber
            linemask = getLinemask(theta,L)
            linemask = linemask / (np.sum(linemask))
            imglinestrength=correlate(Img,linemask)

            imglinestrength = imglinestrength - avgresponse
            lineResponzes[:,:, m] = imglinestrength

        maxlinestrength = np.amax(lineResponzes, axis = 2)
        R = maxlinestrength
        R = standardize(R, Mask, 0)
        features = features + R

    ResultImg = features/(1+len(np.arange(1, winSize, step)))
    ResultImg[ResultImg < 0] = 0

    ResultImg = restoration.denoise_bilateral(ResultImg, sigma_range=0.3, sigma_spatial=15)



    # # LineDetectorImg = float2Uint(SegmentedImg)
    thresh1, Img_BW0 = cv2.threshold(np.float32(ResultImg), threshold, 1, cv2.THRESH_BINARY)


    # Img_BW0 = cv2.adaptiveThreshold(LineDetectorImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                 cv2.THRESH_BINARY, 25, -6) #75, -10;  #95, -15 #55, -10


    return Img_BW0, ResultImg




def lineDetector2(Img_green, Mask, kernelSize = 15):
    Img_green_reverse0 = 255 - Img_green


    Img = im2double(Img_green_reverse0)
    height, width = Img.shape

    winSize = kernelSize #15
    step = 2

    features = standardize(Img, Mask, 0)

    filterNumber = 12
    for i in range(1, winSize+1, step):
        L = i
        # R = get_lineresponse(DilatedImg, winSize, L)
        avgresponse = cv2.blur(Img, ksize=(winSize, winSize))
        lineResponzes = np.ndarray((height, width, filterNumber))

        for m in range(0, filterNumber):
            theta = m * 180/filterNumber
            linemask = getLinemask(theta,L)
            linemask = linemask / (np.sum(linemask))
            imglinestrength=correlate(Img,linemask)

            imglinestrength = imglinestrength - avgresponse
            lineResponzes[:,:, m] = imglinestrength

        maxlinestrength = np.amax(lineResponzes, axis = 2)
        R = maxlinestrength
        R = standardize(R, Mask, 0)
        features = features + R

    ResultImg = features/(1+len(np.arange(1, winSize, step)))
    ResultImg[ResultImg < 0] = 0
    ResultImg = float2Uint(ResultImg)
    ResultImg = cv2.medianBlur(ResultImg, 7)
    # ResultImg = cv2.GaussianBlur(ResultImg, (3,3), 1)
    # ResultImg = restoration.denoise_bilateral(ResultImg, sigma_range=0.3, sigma_spatial=15,  multichannel=False)

    threshold = 0.75
    maskPixCnt = np.count_nonzero(Mask)
    hist,bins = np.histogram(ResultImg.ravel(), 50)
    for i in range(len(hist), 0, -1):
        ratio = np.sum(hist[i:]) / maskPixCnt
        if ratio >= 0.1:
            threshold = bins[i]
            break

    # ResultImg = cv2.medianBlur(ResultImg, 3)
    Img_BW0 = ResultImg>= threshold

    return Img_BW0, ResultImg



def lineDetector_CR(ProfileIntensity, Mask, kernelSize = 15):


    Img = im2double(ProfileIntensity)
    height, width = Img.shape

    winSize = kernelSize
    step = 2

    features = standardize(Img, Mask, 0)

    filterNumber = 17
    for i in range(1, winSize+1, step):
        L = i
        # R = get_lineresponse(DilatedImg, winSize, L)
        avgresponse = cv2.blur(Img, ksize=(winSize, winSize))
        lineResponzes = np.ndarray((height, width, filterNumber))

        for m in range(0, filterNumber):
            theta = m * 180/filterNumber
            linemask = getLinemask(theta,L)
            linemask = linemask / (np.sum(linemask))
            imglinestrength=correlate(Img,linemask)

            imglinestrength = imglinestrength - avgresponse
            lineResponzes[:,:, m] = imglinestrength

        maxlinestrength = np.amax(lineResponzes, axis = 2)
        R = maxlinestrength
        R = standardize(R, Mask, 0)
        features = features + R

    ResultImg = features/(1+len(np.arange(1, winSize, step)))
    ResultImg[ResultImg < 0] = 0
    # ResultImg = cv2.GaussianBlur(ResultImg, (3,3), 1)

    # ResultImg = restoration.denoise_bilateral(ResultImg, sigma_range=0.3, sigma_spatial=15,  multichannel=False)

    threshold = 0.75
    hist,bins = np.histogram(ResultImg.ravel(), 100)
    for i in range(len(hist), 0, -1):
        ratio = np.sum(hist[i:]) / np.count_nonzero(Mask)
        if ratio >= 0.1:
            threshold = bins[i]
            break

    Img_BW0 = ResultImg>= threshold

    return Img_BW0, ResultImg