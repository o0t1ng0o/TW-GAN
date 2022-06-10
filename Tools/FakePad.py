

from __future__ import division

import cv2
import numpy as np
from skimage import morphology
np.seterr(divide='ignore', invalid='ignore')

"""This is the profiled code, very fast, takes 0.25s"""
def fakePad(Image, Mask, iterations=50):
    """
    add an extra padding around the front mask
    :param Image:
    :param Mask:
    :param iterations:
    :return: DilatedImg
    """

    if len(Image.shape) == 3: ##for RGB Image
        """for RGB Images"""

        Mask0 = Mask.copy()
        height, width = Mask0.shape[:2]
        Mask0[0, :] = 0  # np.zeros(width)
        Mask0[-1, :] = 0  # np.zeros(width)
        Mask0[:, 0] = 0  # np.zeros(height)
        Mask0[:, -1] = 0  # np.zeros(height)

        # Erodes the mask to avoid weird region near the border.
        structureElement1 = morphology.disk(5)
        Mask0 = cv2.morphologyEx(Mask0, cv2.MORPH_ERODE, structureElement1, iterations=1)

        # DilatedImg = Img_green_reverse * Mask
        DilatedImg = cv2.bitwise_and(Image, Image, mask=Mask0)
        OldMask = Mask0.copy()

        filter = np.ones((3, 3))
        filterRows, filterCols = np.where(filter > 0)
        filterRows = filterRows - 1
        filterCols = filterCols - 1

        structureElement2 = morphology.diamond(1)
        for i in range(0, iterations):
            NewMask = cv2.morphologyEx(OldMask, cv2.MORPH_DILATE, structureElement2, iterations=1)
            pixelIndex = np.where(NewMask - OldMask)  # [rows, cols]
            imgValues = np.zeros((len(pixelIndex[0]), len(filterRows), 3))
            for k in range(len(filterRows)):
                filterRowIndexes = pixelIndex[0] - filterRows[k]
                filterColIndexes = pixelIndex[1] - filterCols[k]

                selectMask0 = np.bitwise_and(np.bitwise_and(filterRowIndexes < height, filterRowIndexes >= 0),
                                             np.bitwise_and(filterColIndexes < width, filterColIndexes >= 0))
                selectMask1 = OldMask[filterRowIndexes[selectMask0], filterColIndexes[selectMask0]] > 0
                selectedPositions = [filterRowIndexes[selectMask0][selectMask1],
                                     filterColIndexes[selectMask0][selectMask1]]
                imgValues[np.arange(len(pixelIndex[0]))[selectMask0][selectMask1], k, :] = DilatedImg[
                                                                                           selectedPositions[0],
                                                                                           selectedPositions[1], :]

            DilatedImg[pixelIndex[0], pixelIndex[1], :] = np.sum(imgValues, axis=1) // np.sum(imgValues > 0, axis=1)

            OldMask = NewMask

        return DilatedImg

    ########################################################################

    else:   #for green channel only
        """for green channel only"""

        Mask0 = Mask.copy()
        height, width = Mask0.shape
        Mask0[0, :] = 0  # np.zeros(width)
        Mask0[-1, :] = 0  # np.zeros(width)
        Mask0[:, 0] = 0  # np.zeros(height)
        Mask0[:, -1] = 0  # np.zeros(height)

        # Erodes the mask to avoid weird region near the border.
        structureElement1 = morphology.disk(5)
        Mask0 = cv2.morphologyEx(Mask0, cv2.MORPH_ERODE, structureElement1, iterations=1)

        # DilatedImg = Img_green_reverse * Mask
        DilatedImg = cv2.bitwise_and(Image, Image, mask=Mask0)

        OldMask = Mask0.copy()

        filter = np.ones((3, 3))
        filterRows, filterCols = np.where(filter > 0)
        filterRows = filterRows - 1
        filterCols = filterCols - 1

        structureElement2 = morphology.diamond(1)
        for i in range(0, iterations):
            NewMask = cv2.morphologyEx(OldMask, cv2.MORPH_DILATE, structureElement2, iterations=1)
            pixelIndex = np.where(NewMask - OldMask)  # [rows, cols]

            imgValues = np.zeros((len(pixelIndex[0]), len(filterRows)))
            for k in range(len(filterRows)):
                filterRowIndexes = pixelIndex[0] - filterRows[k]
                filterColIndexes = pixelIndex[1] - filterCols[k]

                selectMask0 = np.bitwise_and(np.bitwise_and(filterRowIndexes < height, filterRowIndexes >= 0),
                                             np.bitwise_and(filterColIndexes < width, filterColIndexes >= 0))
                selectMask1 = OldMask[filterRowIndexes[selectMask0], filterColIndexes[selectMask0]] > 0
                selectedPositions = [filterRowIndexes[selectMask0][selectMask1], filterColIndexes[selectMask0][selectMask1]]
                imgValues[np.arange(len(pixelIndex[0]))[selectMask0][selectMask1], k] = DilatedImg[selectedPositions[0], selectedPositions[1]]

            DilatedImg[pixelIndex[0], pixelIndex[1]] = np.sum(imgValues, axis=1) / np.sum(imgValues > 0, axis=1)

            OldMask = NewMask

        return DilatedImg


