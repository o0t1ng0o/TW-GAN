

import numpy as np
from skimage import morphology, measure
from Tools.Remove_small_holes import remove_small_holes
import scipy.ndimage.morphology as scipyMorphology


def binaryPostProcessing(BinaryImage, removeArea):
    """
    Post process the binary segmentation
    :param BinaryImage:
    :param removeArea:
    :return: Img_BW
    """

    BinaryImage[BinaryImage > 0] = 1

    ###9s
    # Img_BW = pymorph.binary(BinaryImage)
    # Img_BW = pymorph.areaopen(Img_BW, removeArea)
    # Img_BW = pymorph.areaclose(Img_BW, 50)
    # Img_BW = np.uint8(Img_BW)

    ###2.5 s
    # Img_BW = np.uint8(BinaryImage)
    # Img_BW = ITK_LabelImage(Img_BW, removeArea)
    # Img_BW[Img_BW >0] = 1

    Img_BW = BinaryImage.copy()
    BinaryImage_Label = measure.label(Img_BW)
    for i, region in enumerate(measure.regionprops(BinaryImage_Label)):
        if region.area < removeArea:
            Img_BW[BinaryImage_Label == i + 1] = 0
        else:
            pass

    Img_BW = morphology.binary_closing(Img_BW, morphology.disk(3))
    Img_BW = remove_small_holes(Img_BW, 50)
    Img_BW = np.uint8(Img_BW)

    return Img_BW


################Three parameters
def binaryPostProcessing3(BinaryImage, removeArea, fillArea):
    """
    Post process the binary image
    :param BinaryImage:
    :param removeArea:
    :param fillArea:
    :return: Img_BW
    """

    BinaryImage[BinaryImage>0]=1

    ####takes 0.9s, result is good
    Img_BW = BinaryImage.copy()
    BinaryImage_Label = measure.label(Img_BW)
    for i, region in enumerate(measure.regionprops(BinaryImage_Label)):
        if region.area < removeArea:
            Img_BW[BinaryImage_Label == i + 1] = 0
        else:
            pass

    # ####takes 0.01s, result is bad
    # temptime = time.time()
    # Img_BW = morphology.remove_small_objects(BinaryImage, removeArea)
    # print "binaryPostProcessing3, ITK_LabelImage time:", time.time() - temptime


    Img_BW = morphology.binary_closing(Img_BW, morphology.square(3))
    # Img_BW = remove_small_holes(Img_BW, fillArea)

    Img_BW_filled = scipyMorphology.binary_fill_holes(Img_BW)
    Img_BW_dif = np.uint8(Img_BW_filled) - np.uint8(Img_BW)
    Img_BW_difLabel = measure.label(Img_BW_dif)
    FilledImg = np.zeros(Img_BW.shape)
    for i, region in enumerate(measure.regionprops(Img_BW_difLabel)):
        if region.area < fillArea:
            FilledImg[Img_BW_difLabel == i + 1] = 1
        else:
            pass
        Img_BW[FilledImg > 0] = 1

    Img_BW = np.uint8(Img_BW)
    return Img_BW


def removeSmallBLobs(BinaryImage, removeArea):
    """
    Post process the binary image
    :param BinaryImage:
    :param removeArea:
    """

    BinaryImage[BinaryImage>0]=1

    ####takes 0.9s, result is good
    Img_BW = BinaryImage.copy()
    BinaryImage_Label = measure.label(Img_BW)
    for i, region in enumerate(measure.regionprops(BinaryImage_Label)):
        if region.area < removeArea:
            Img_BW[BinaryImage_Label == i + 1] = 0
        else:
            pass
    return np.uint8(Img_BW)