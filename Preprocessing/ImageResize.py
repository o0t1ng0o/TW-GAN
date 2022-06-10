

import cv2
import numpy as np
from skimage import measure

def imageResize(Image, downsizeRatio):

    ##This program resize the original image
    ##Input: original image and downsizeRatio (user defined parameter: 0.75, 0.5 or 0.2)
    ##Output: the resized image according to the given ratio

    if  downsizeRatio < 1:#len(ImgFileList)
        ImgResized = cv2.resize(Image, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
    else:
        ImgResized = Image

    ImgResized = np.uint8(ImgResized)
    return ImgResized


def creatMask(Image, threshold = 10):
    ##This program try to creat the mask for the filed-of-view
    ##Input original image (RGB or green channel), threshold (user set parameter, default 10)
    ##Output: the filed-of-view mask

    if len(Image.shape) == 3: ##RGB image
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Mask0 = gray >= threshold

    else:  #for green channel image
        Mask0 = Image >= threshold


    # ######get the largest blob, this takes 0.18s
    cvVersion = int(cv2.__version__.split('.')[0])

    Mask0 = np.uint8(Mask0)
    if cvVersion == 2:
        contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    Mask = np.zeros(Image.shape[:2], dtype=np.uint8)
    cv2.drawContours(Mask, contours, max_index, 1, -1)

    ResultImg = Image.copy()
    if len(Image.shape) == 3:
        ResultImg[Mask ==0] = (255,255,255)
    else:
        ResultImg[Mask==0] = 255

    return ResultImg, Mask


def cropImage_bak(Image, Mask):
    Image = Image.copy()
    Mask = Mask.copy()

    leftLimit, rightLimit, upperLimit, lowerLimit = GetLimit(Mask)



    if len(Image.shape) == 3:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit, :]
        MaskCropped = Mask[upperLimit:lowerLimit, leftLimit:rightLimit]

        ImgCropped[:20, :, :] = 0
        ImgCropped[-20:, :, :] = 0
        ImgCropped[:, :20, :] = 0
        ImgCropped[:, -20:, :] = 0
        MaskCropped[:20, :] = 0
        MaskCropped[-20:, :] = 0
        MaskCropped[:, :20] = 0
        MaskCropped[:, -20:] = 0
    else: #len(Image.shape) == 2:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit]
        MaskCropped = Mask[upperLimit:lowerLimit, leftLimit:rightLimit]
        ImgCropped[:20, :] = 0
        ImgCropped[-20:, :] = 0
        ImgCropped[:, :20] = 0
        ImgCropped[:, -20:] = 0
        MaskCropped[:20, :] = 0
        MaskCropped[-20:, :] = 0
        MaskCropped[:, :20] = 0
        MaskCropped[:, -20:] = 0


    cropLimit = [upperLimit, lowerLimit, leftLimit, rightLimit]

    return ImgCropped, MaskCropped, cropLimit



########################################################
###new function to get the limit for cropping.
###try to get higher speed than np.where, but not working.

def getLimit(Mask):

    Mask1 = Mask > 0
    colSums = np.sum(Mask1, axis=1)
    rowSums = np.sum(Mask1, axis=0)
    maxColSum = np.max(colSums)
    maxRowSum = np.max(rowSums)

    colList = np.where(colSums >= 0.01*maxColSum)[0]
    rowList = np.where(rowSums >= 0.01*maxRowSum)[0]

    leftLimit0 = np.min(rowList)
    rightLimit0 = np.max(rowList)
    upperLimit0 = np.min(colList)
    lowerLimit0 = np.max(colList)

    margin = 50
    leftLimit = np.clip(leftLimit0-margin, 0, Mask.shape[1])
    rightLimit = np.clip(rightLimit0+margin, 0, Mask.shape[1])
    upperLimit = np.clip(upperLimit0 - margin, 0, Mask.shape[0])
    lowerLimit = np.clip(lowerLimit0 + margin, 0, Mask.shape[0])


    return leftLimit, rightLimit, upperLimit, lowerLimit





def cropImage(Image, Mask):
    ##This program will crop the filed of view based on the mask
    ##Input: orginal image, origimal Mask  (the image needs to be RGB resized image)
    ##Output: Cropped image, Cropped Mask, the cropping limit

    height, width = Image.shape[:2]

    rowsMask0, colsMask0 = np.where(Mask > 0)
    minColIndex0, maxColIndex0 = np.argmin(colsMask0), np.argmax(colsMask0)
    minCol, maxCol = colsMask0[minColIndex0], colsMask0[maxColIndex0]

    minRowIndex0, maxRowIndex0 = np.argmin(rowsMask0), np.argmax(rowsMask0)
    minRow, maxRow = rowsMask0[minRowIndex0], rowsMask0[maxRowIndex0]

    upperLimit = np.maximum(0, minRow - 50)   #20
    lowerLimit = np.minimum(maxRow + 50, height)   #20
    leftLimit = np.maximum(0, minCol - 50)   #lowerLimit = np.minimum(maxCol + 50, width)   #20
    rightLimit = np.minimum(maxCol + 50, width)

    if len(Image.shape) == 3:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit, :]
        MaskCropped = Mask[upperLimit:lowerLimit, leftLimit:rightLimit]

        ImgCropped[:20, :, :] = 0
        ImgCropped[-20:, :, :] = 0
        ImgCropped[:, :20, :] = 0
        ImgCropped[:, -20:, :] = 0
        MaskCropped[:20, :] = 0
        MaskCropped[-20:, :] = 0
        MaskCropped[:, :20] = 0
        MaskCropped[:, -20:] = 0
    elif len(Image.shape) == 2:
        ImgCropped = Image[upperLimit:lowerLimit, leftLimit:rightLimit]
        MaskCropped = Mask[upperLimit:lowerLimit, leftLimit:rightLimit]
        ImgCropped[:20, :] = 0
        ImgCropped[-20:, :] = 0
        ImgCropped[:, :20] = 0
        ImgCropped[:, -20:] = 0
        MaskCropped[:20, :] = 0
        MaskCropped[-20:, :] = 0
        MaskCropped[:, :20] = 0
        MaskCropped[:, -20:] = 0
    else:
        pass


    cropLimit = [upperLimit, lowerLimit, leftLimit, rightLimit]

    return ImgCropped, MaskCropped, cropLimit
