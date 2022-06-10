

from __future__ import division
import numpy as np
import cv2

def standardize(img,mask,wsize):
    """
    Convert the image values to standard images.
    :param img:
    :param mask:
    :param wsize:
    :return:
    """

    if  wsize == 0:
        simg=globalstandardize(img,mask)
    else:
        img[mask == 0]=0
        img_mean=cv2.blur(img, ksize=wsize)
        img_squared_mean = cv2.blur(img*img, ksize=wsize)
        img_std = np.sqrt(img_squared_mean - img_mean*img_mean)
        simg=(img - img_mean) / img_std
        simg[img_std == 0]=0
        simg[mask == 0]=0
    return simg

def globalstandardize(img,mask):

    usedpixels = np.double(img[mask == 1])
    m=np.mean(usedpixels)
    s=np.std(usedpixels)
    simg=np.zeros(img.shape)
    simg[mask == 1]=(usedpixels - m) / s
    return simg

def getmean(x):
    usedx=x[x != 0]
    m=np.mean(usedx)
    return m

def getstd(x):
    usedx=x[x != 0]
    s=np.std(usedx)
    return s