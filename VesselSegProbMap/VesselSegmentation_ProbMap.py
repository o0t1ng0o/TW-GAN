import cv2
import numpy as np
from skimage import measure, morphology
import os
import time
import pandas as pd

import matplotlib.pyplot as plt
from Tools.BGR2RGB import BGR2RGB, RGB2BGR
# from Tools.SortFolder import natsort
import natsort
from Segmentation.LineDetector import lineDetector2
from Tools.BinaryPostProcessing import binaryPostProcessing3
from Segmentation.HysteresisThresholding import hysteresisThresholding
from Preprocessing.ImageResize import creatMask, imageResize, cropImage
from Preprocessing.IlluminationCorrection import illuminationCorrection
from Segmentation.TwoScalesGabor import twoScalesGabor
from Tools.FakePad import fakePad
from Tools.Im2Double import im2double


def VesselProMap(path):
    """Load Images """
    ImgNumber = 0
    
    folder = path
    ImgList0 = os.listdir(folder)
    ImgList0 = natsort.natsorted(ImgList0)
    ImgList = [x  for x in ImgList0 if x.__contains__('.jpg') or x.__contains__('.JPG') or x.__contains__('.tif') or x.__contains__('.png')]
    imgName = os.path.join(folder, ImgList[0])
    Img0 = cv2.imread(imgName)
    _, TempMask = creatMask(Img0, threshold=10)
#    downsizeRatio = 1000./np.maximum(Img0.shape[0], Img0.shape[1])
    ImgResized = Img0
    ProImg = np.zeros((len(ImgList), 2, ImgResized.shape[0], ImgResized.shape[1]), np.float32)
    for i in range(len(ImgList)):
        imgName = os.path.join(folder, ImgList[i])
        Img0 = cv2.imread(imgName)
        
        #############################################################
        """Preprocessing:  Cropping, Resizing, and Illumination Normalization"""
        
        
        
        _, TempMask = creatMask(Img0, threshold=10)
#        downsizeRatio = 1000./np.maximum(Img0.shape[0], Img0.shape[1])
        ImgResized = Img0
        MaskResized = TempMask
        Img = ImgResized  ##Replace the Image with Cropped Image
        Mask = MaskResized  ##Replace the Mask with cropped mask
        height, width = Img.shape[:2]
        
        # ImgShow = Img.copy()
        
        
        IllumImage = illuminationCorrection(Img, kernel_size=35, Mask=Mask)
        IllumGreen = IllumImage[:,:,1]
        
#        print(i)
        
        
        #############################################################
        """Vessel Segmentation and Skeletonization"""
        
        
        
        ##########Method 1#########################
        _, VesselProbImg1 = lineDetector2(IllumGreen, Mask,  kernelSize = 15)
        Img_BW1 = hysteresisThresholding(VesselProbImg1, Mask)
        Img_BW1= binaryPostProcessing3(Img_BW1, removeArea=300, fillArea=100)
        
        
        ############Method 2################################
        
        
        IllumImage_green = IllumImage[:, :, 1]
        Img_green_reverse0 = 255 - IllumImage_green
        Img_green_reverse0 = cv2.medianBlur(Img_green_reverse0, ksize=5)
        Img_green_reverse = im2double(Img_green_reverse0)
        
        Img_BW2, VesselProbImg2 = twoScalesGabor(Img_green_reverse, Mask)
        Img_BW2 = cv2.bitwise_and(Img_BW2, Img_BW2, mask=Mask)
        
        
        ##################################################################################################
#        print("End of Image Processing >>>", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#        plt.figure()
#        Images = [BGR2RGB(Img),  IllumGreen, VesselProbImg1, VesselProbImg2, Img_BW1, Img_BW2, ]
#        Titles = [ 'ImgShow', 'IllumGreen', 'VesselProbImg1', 'VesselProbImg2', 'Img_BW1', 'Img_BW2', '']
        
#        for i in range(0, len(Images)):
#            plt.subplot(2, 4, i + 1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
#        plt.show()
        
        VesselProbImg1 = VesselProbImg1/255.
        VesselProbImg2 = VesselProbImg2/255.
        ProImg[i,0,:,:] = VesselProbImg1
        ProImg[i,1,:,:] = VesselProbImg2
        
    return ProImg
        
#path = r'E:\code_old\code\data\INSPIRE-AVR\Image'
#path = r'E:\code_old\code\data\AV_DRIVE\test\images'
#ProImg = VesselProMap(path)
