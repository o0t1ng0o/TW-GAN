# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import cv2
from Tools.ImageResize import creatMask
from Tools.data_augmentation import data_aug5, data_aug7, data_aug9
from lib.Utils import *
from models.network import TWGAN_Net
from Tools.FakePad import fakePad
from sklearn.metrics import roc_auc_score
from VesselSegProbMap.VesselSegmentation_ProbMap import VesselProMap
import natsort
from DRIVE_Evalution import Evalution_drive
from HRF_Evalution import Evalution_HRF
from Tools.AVclassifiationMetrics import AVclassifiationMetrics_skeletonPixles
from Tools.centerline_evaluation import centerline_eval, getFolds

def get_patch_trad_5(batch_size, patch_size, train_data1, label_data, label_data_fake, label_data_fake2, label_data_fake3, patchsize1=96, patchsize2=128):
    data1 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake2 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake3 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    # z = np.random.randint(0,20)
    # patchsize1 = 96
    # patchsize2 = 128
    for j in range(batch_size):

        random_size = np.random.randint(0, 3)
        z = np.random.randint(0, 20)
        choice = np.random.randint(0, 5)
        # PatchNum = np.random.randint(0,train_data.shape[2])
        if random_size == 0:
            x = np.random.randint(0, train_data1.shape[2] - patch_size + 1)
            y = np.random.randint(0, train_data1.shape[3] - patch_size + 1)
            data_mat_1 = train_data1[z, :, x:x + patch_size, y:y + patch_size]
            label_mat = label_data[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake = label_data_fake[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake2 = label_data_fake2[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake3 = label_data_fake3[z, :, x:x + patch_size, y:y + patch_size]
            # label[j,:,:,:] = label_data[z,:,PatchNum,:,:]
        elif random_size == 1:
            x = np.random.randint(0, train_data1.shape[2] - patchsize1 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize1 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        else:
            x = np.random.randint(0, train_data1.shape[2] - patchsize2 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize2 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3 = data_aug5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, choice)
        data1[j, :, :, :] = data_mat_1
        label[j, :, :, :] = label_mat
        label_fake[j, :, :, :] = label_mat_fake
        label_fake2[j, :, :, :] = label_mat_fake2
        label_fake3[j, :, :, :] = label_mat_fake3
    return data1, label, label_fake, label_fake2, label_fake3


def get_patch_trad_7(batch_size, patch_size, train_data1, label_data, label_data_fake, label_data_fake2, label_data_fake3, label_data_fake4, label_data_fake5, patchsize1=96, patchsize2=128):
    data1 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake2 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake3 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake4 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake5 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    # z = np.random.randint(0,20)
    # patchsize1 = 96
    # patchsize2 = 128
    for j in range(batch_size):

        random_size = np.random.randint(0, 3)
        z = np.random.randint(0, 20)
        choice = np.random.randint(0, 5)
        # PatchNum = np.random.randint(0,train_data.shape[2])
        if random_size == 0:
            x = np.random.randint(0, train_data1.shape[2] - patch_size + 1)
            y = np.random.randint(0, train_data1.shape[3] - patch_size + 1)
            data_mat_1 = train_data1[z, :, x:x + patch_size, y:y + patch_size]
            label_mat = label_data[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake = label_data_fake[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake2 = label_data_fake2[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake3 = label_data_fake3[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake4 = label_data_fake4[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake5 = label_data_fake5[z, :, x:x + patch_size, y:y + patch_size]
            # label[j,:,:,:] = label_data[z,:,PatchNum,:,:]
        elif random_size == 1:
            x = np.random.randint(0, train_data1.shape[2] - patchsize1 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize1 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake4 = np.transpose(cv2.resize(np.transpose(label_data_fake4[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake5 = np.transpose(cv2.resize(np.transpose(label_data_fake5[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        else:
            x = np.random.randint(0, train_data1.shape[2] - patchsize2 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize2 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake4 = np.transpose(cv2.resize(np.transpose(label_data_fake4[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake5 = np.transpose(cv2.resize(np.transpose(label_data_fake5[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5 = \
                data_aug7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, choice)
        data1[j, :, :, :] = data_mat_1
        label[j, :, :, :] = label_mat
        label_fake[j, :, :, :] = label_mat_fake
        label_fake2[j, :, :, :] = label_mat_fake2
        label_fake3[j, :, :, :] = label_mat_fake3
        label_fake4[j, :, :, :] = label_mat_fake4
        label_fake5[j, :, :, :] = label_mat_fake5
    return data1, label, label_fake, label_fake2, label_fake3, label_fake4, label_fake5


def get_patch_trad_9(batch_size, patch_size, train_data1, label_data, label_data_fake, label_data_fake2, label_data_fake3, label_data_fake4, label_data_fake5, label_data_fake6, label_data_fake7, patchsize1=96, patchsize2=128):
    data1 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake2 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake3 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake4 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake5 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake6 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_fake7 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    # z = np.random.randint(0,20)
    # patchsize1 = 96
    # patchsize2 = 128
    for j in range(batch_size):

        random_size = np.random.randint(0, 3)
        z = np.random.randint(0, 20)
        choice = np.random.randint(0, 5)
        # PatchNum = np.random.randint(0,train_data.shape[2])
        if random_size == 0:
            x = np.random.randint(0, train_data1.shape[2] - patch_size + 1)
            y = np.random.randint(0, train_data1.shape[3] - patch_size + 1)
            data_mat_1 = train_data1[z, :, x:x + patch_size, y:y + patch_size]
            label_mat = label_data[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake = label_data_fake[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake2 = label_data_fake2[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake3 = label_data_fake3[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake4 = label_data_fake4[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake5 = label_data_fake5[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake6 = label_data_fake6[z, :, x:x + patch_size, y:y + patch_size]
            label_mat_fake7 = label_data_fake7[z, :, x:x + patch_size, y:y + patch_size]
            # label[j,:,:,:] = label_data[z,:,PatchNum,:,:]
        elif random_size == 1:
            x = np.random.randint(0, train_data1.shape[2] - patchsize1 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize1 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake4 = np.transpose(cv2.resize(np.transpose(label_data_fake4[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake5 = np.transpose(cv2.resize(np.transpose(label_data_fake5[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake6 = np.transpose(cv2.resize(np.transpose(label_data_fake6[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake7 = np.transpose(cv2.resize(np.transpose(label_data_fake7[z, :, x:x + patchsize1, y:y + patchsize1], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        else:
            x = np.random.randint(0, train_data1.shape[2] - patchsize2 + 1)
            y = np.random.randint(0, train_data1.shape[3] - patchsize2 + 1)
            data_mat_1 = np.transpose(
                cv2.resize(np.transpose(train_data1[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat = np.transpose(
                cv2.resize(np.transpose(label_data[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake = np.transpose(
                cv2.resize(np.transpose(label_data_fake[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake2 = np.transpose(
                cv2.resize(np.transpose(label_data_fake2[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),
                           (patch_size, patch_size)), (2, 0, 1))
            label_mat_fake3 = np.transpose(cv2.resize(np.transpose(label_data_fake3[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake4 = np.transpose(cv2.resize(np.transpose(label_data_fake4[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake5 = np.transpose(cv2.resize(np.transpose(label_data_fake5[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake6 = np.transpose(cv2.resize(np.transpose(label_data_fake6[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
            label_mat_fake7 = np.transpose(cv2.resize(np.transpose(label_data_fake7[z, :, x:x + patchsize2, y:y + patchsize2], (1, 2, 0)),(patch_size, patch_size)), (2, 0, 1))
        
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7  = \
                data_aug9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7, choice)
        data1[j, :, :, :] = data_mat_1
        label[j, :, :, :] = label_mat
        label_fake[j, :, :, :] = label_mat_fake
        label_fake2[j, :, :, :] = label_mat_fake2
        label_fake3[j, :, :, :] = label_mat_fake3
        label_fake4[j, :, :, :] = label_mat_fake4
        label_fake5[j, :, :, :] = label_mat_fake5
        label_fake6[j, :, :, :] = label_mat_fake6
        label_fake7[j, :, :, :] = label_mat_fake7
    return data1, label, label_fake, label_fake2, label_fake3, label_fake4, label_fake5, label_fake6, label_fake7

def Dataloader_DRIVE(path, path_gt_fake, path_centerness, dilation_list = [], overlap=0, isNormalize=False):
    # isNormalize: whether normalize the image of path_gt_fake
    #    (1) True,  [0,255] -> [0,1].
    #    (2) False, not treatment.
    # overlap : how to treat the overlapping part of vessel
    # overlap == 0 : artery = 1, vein = 1
    # overlap == 1 : artery = 1, vein = 0
    # overlap == 2 : artery = 0, vein = 1
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    centPath = path + "dilation_"

    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)

    
    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    
    Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
    Img0 = np.float32(Img0/255.)
    
    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    Label_fake = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    Label_center = []
    Label_dilation=[]
    for i in range(len(dilation_list)):
        Label_center.append(np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32))
        Label_dilation.append(np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32))

    for i in range(0,20):
        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        # fake label

        LabelArtery_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)

        Img0 = cv2.imread(ImgPath + str(i+21) + '_training.tif')
        Label0 = cv2.imread(LabelPath + str(i+21) + '_training.png')
        Label0_fake = cv2.imread(path_gt_fake + str(i + 21) + '_training.png')

        idx = 0
        for dil in dilation_list:

            Labelcenter_artery= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'a_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
            Labelcenter_vein  = cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'v_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
            Labelcenter_vessel= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'ves_' + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)

            Labelcenter_artery = Labelcenter_artery / 255.0
            Labelcenter_vein   = Labelcenter_vein   / 255.0
            Labelcenter_vessel = Labelcenter_vessel / 255.0

            Label_center[idx][i,0,:,:] = Labelcenter_artery
            Label_center[idx][i,1,:,:] = Labelcenter_vein
            Label_center[idx][i,2,:,:] = Labelcenter_vessel

            # load dilated images
            if dil == 0 :
                Labeldil_artery= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'diskless_a_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
                Labeldil_vein  = cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'diskless_v_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
                Labeldil_vessel= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'diskless_ves_' + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
            else:
                Labeldil_artery= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_a_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
                Labeldil_vein  = cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_v_'   + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)
                Labeldil_vessel= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_ves_' + str(i+21) + '_training.png', cv2.IMREAD_GRAYSCALE)

            Labeldil_artery = Labeldil_artery / 255.0
            Labeldil_vein   = Labeldil_vein   / 255.0
            Labeldil_vessel = Labeldil_vessel / 255.0

            Label_dilation[idx][i,0,:,:] = Labeldil_artery
            Label_dilation[idx][i,1,:,:] = Labeldil_vein  
            Label_dilation[idx][i,2,:,:] = Labeldil_vessel

            idx += 1

            
        TempImg, TempMask = creatMask(Img0, threshold=10)
        #ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
        ImgCropped = Img0
        #ImgIllCropped, MaskCropped, cropLimit = cropImage(IllumImage, TempMask)
        # BGR
        if overlap == 0 or overlap == 1:
            LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
        else:
            LabelArtery[(Label0[:,:,2]==255)] = 1
        LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0

        if overlap == 0 or overlap == 2:
            LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        else:
            LabelVein[(Label0[:,:,0]==255)] = 1
        LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        # fake label
        if isNormalize:
            LabelArtery_fake = Label0_fake[:,:,2]/255
            LabelVein_fake =  Label0_fake[:,:,0]/255
            LabelVessel_fake = Label0_fake[:, :, 1] / 255
        else:
            LabelArtery_fake[(Label0_fake[:,:,2]==255)&(Label0_fake[:,:,1]==255)&(Label0_fake[:,:,0]==255)] = 0
            if overlap == 0 or overlap == 1:
                LabelArtery_fake[(Label0_fake[:,:,2]==255)|(Label0_fake[:,:,1]==255)] = 1
            else:
                LabelArtery_fake[(Label0_fake[:,:,2]==255)] = 1

            if overlap == 0 or overlap == 2:
                LabelVein_fake[(Label0_fake[:,:,1]==255)|(Label0_fake[:,:,0]==255)] = 1
            else:
                LabelVein_fake[(Label0_fake[:,:,0]==255)] = 1
            LabelVein_fake[(Label0_fake[:,:,2]==255)&(Label0_fake[:,:,1]==255)&(Label0_fake[:,:,0]==255)] = 0
            LabelVessel_fake[(Label0_fake[:,:,2]==255)|(Label0_fake[:,:,1]==255)|(Label0_fake[:,:,0]==255)] = 1

        ImgCropped = cv2.cvtColor(ImgCropped, cv2.COLOR_BGR2RGB)
        ImgCropped = np.float32(ImgCropped/255.)
        Img[i,:,:,:] = np.transpose(ImgCropped,(2,0,1))

        Label[i,0,:,:] = LabelArtery
        Label[i,1,:,:] = LabelVein
        Label[i,2,:,:] = LabelVessel
        # fake
        Label_fake[i,0,:,:] = LabelArtery_fake
        Label_fake[i,1,:,:] = LabelVein_fake
        Label_fake[i,2,:,:] = LabelVessel_fake

    Img = Normalize(Img)

    return Img,Label,Label_fake, Label_center, Label_dilation


def Dataloader_HRF(path, path_gt_fake, path_centerness, dilation_list = [], overlap=0, k_fold_idx=0, k_fold=0):
    # overlap : how to treat the overlapping part of vessel
    # overlap == 0 : artery = 1, vein = 1
    # overlap == 1 : artery = 1, vein = 0
    # overlap == 2 : artery = 0, vein = 1

    is_AFIO = path.split('/')[2] == 'AFIO'

    ImgPath = path + "images/"
    LabelPath = path + "ArteryVein_0410_final/"

    ImgDir , LabelDir = getFolds(ImgPath, LabelPath, k_fold_idx, k_fold)
    

    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])

    Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
    Img0 = np.float32(Img0 / 255.)

    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    Label_fake = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    ImgDir = natsort.natsorted(ImgDir)
    ImgList = [x  for x in ImgDir if x.__contains__('.jpg') or x.__contains__('.JPG') or x.__contains__('.tif') or x.__contains__('.png')]

    LabelDir = natsort.natsorted(LabelDir)
    LabelList = [x for x in LabelDir if
               x.__contains__('.jpg') or x.__contains__('.JPG') or x.__contains__('.tif') or x.__contains__('.png')]

    Label_center = []
    Label_dilation=[]
    for i in range(len(dilation_list)):
        Label_center.append(np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32))
        Label_dilation.append(np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32))


    for i in range(len(ImgList)):
        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        # fake label

        LabelArtery_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel_fake = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)

        Img0 =          cv2.imread(ImgPath      + ImgList[i])
        Label0 =        cv2.imread(LabelPath    + LabelList[i])
        Label0_fake =   cv2.imread(path_gt_fake + LabelList[i])

        idx = 0
        for dil in dilation_list:
            if is_AFIO:
                ImgList[i] = ImgList[i].replace('JPG','png')
            
            Labelcenter_artery= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'a_'   + ImgList[i], cv2.IMREAD_GRAYSCALE)
            Labelcenter_vein  = cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'v_'   + ImgList[i], cv2.IMREAD_GRAYSCALE)
            Labelcenter_vessel= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'ves_' + ImgList[i], cv2.IMREAD_GRAYSCALE)

            Labelcenter_artery = Labelcenter_artery / 255.0
            Labelcenter_vein   = Labelcenter_vein   / 255.0
            Labelcenter_vessel = Labelcenter_vessel / 255.0

            Label_center[idx][i,0,:,:] = Labelcenter_artery
            Label_center[idx][i,1,:,:] = Labelcenter_vein
            Label_center[idx][i,2,:,:] = Labelcenter_vessel

            # load dilated images
            Labeldil_artery= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_a_'   + ImgList[i], cv2.IMREAD_GRAYSCALE)
            Labeldil_vein  = cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_v_'   + ImgList[i], cv2.IMREAD_GRAYSCALE)
            Labeldil_vessel= cv2.imread(path_centerness + 'dilation_' + str(dil) +'/' + 'dil_ves_' + ImgList[i], cv2.IMREAD_GRAYSCALE)

            Labeldil_artery = Labeldil_artery / 255.0
            Labeldil_vein   = Labeldil_vein   / 255.0
            Labeldil_vessel = Labeldil_vessel / 255.0

            Label_dilation[idx][i,0,:,:] = Labeldil_artery
            Label_dilation[idx][i,1,:,:] = Labeldil_vein  
            Label_dilation[idx][i,2,:,:] = Labeldil_vessel

            idx += 1


        TempImg, TempMask = creatMask(Img0, threshold=10)
        # ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
        ImgCropped = Img0
        # ImgIllCropped, MaskCropped, cropLimit = cropImage(IllumImage, TempMask)
        # BGR
        if overlap == 0 or overlap == 1:
            LabelArtery[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255)] = 1
        else:
            LabelArtery[(Label0[:, :, 2] == 255)] = 1
        LabelArtery[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        if overlap == 0 or overlap == 2:
            LabelVein[(Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        else:
            LabelVein[(Label0[:, :, 0] == 255)] = 1
        LabelVein[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0
        LabelVessel[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        # fake label
        LabelArtery_fake[
            (Label0_fake[:, :, 2] == 255) & (Label0_fake[:, :, 1] == 255) & (Label0_fake[:, :, 0] == 255)] = 0
        if overlap == 0 or overlap == 1:
            LabelArtery_fake[(Label0_fake[:, :, 2] == 255) | (Label0_fake[:, :, 1] == 255)] = 1
        else:
            LabelArtery_fake[(Label0_fake[:, :, 2] == 255)] = 1

        if overlap == 0 or overlap == 2:
            LabelVein_fake[(Label0_fake[:, :, 1] == 255) | (Label0_fake[:, :, 0] == 255)] = 1
        else:
            LabelVein_fake[(Label0_fake[:, :, 0] == 255)] = 1
        LabelVein_fake[
            (Label0_fake[:, :, 2] == 255) & (Label0_fake[:, :, 1] == 255) & (Label0_fake[:, :, 0] == 255)] = 0
        LabelVessel_fake[
            (Label0_fake[:, :, 2] == 255) | (Label0_fake[:, :, 1] == 255) | (Label0_fake[:, :, 0] == 255)] = 1

        ImgCropped = cv2.cvtColor(ImgCropped, cv2.COLOR_BGR2RGB)
        ImgCropped = np.float32(ImgCropped / 255.)
        Img[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1))

        Label[i, 0, :, :] = LabelArtery
        Label[i, 1, :, :] = LabelVein
        Label[i, 2, :, :] = LabelVessel
        # fake
        Label_fake[i, 0, :, :] = LabelArtery_fake
        Label_fake[i, 1, :, :] = LabelVein_fake
        Label_fake[i, 2, :, :] = LabelVessel_fake

    Img = Normalize(Img)

    return Img, Label, Label_fake, Label_center, Label_dilation



def modelEvalution_inspire(i, net, savePath, loss_all, input_ch=3, use_output_block=False, use_spade=False, use_cuda=False, config=None, strict_mode=False):
    # path for images to save
    # inspire_path = os.path.join(savePath, 'inspire')
    # metrics_file_path = os.path.join(savePath, 'metrics_inspire.txt')
    inspire_path = os.path.join(savePath, 'inspire')
    metrics_file_path = os.path.join(savePath, 'metrics_inspire.txt')
    if not os.path.exists(inspire_path):
        os.mkdir(inspire_path)

    promap2 = np.zeros((40, 3, 342, 400), np.float32)
    labelmap = np.zeros((40, 3, 342, 400), np.float32)

    arterypredall = np.zeros((40, 1, 342, 400), np.float32)
    veinpredall = np.zeros((40, 1, 342, 400), np.float32)
    vesselpredall = np.zeros((40, 1, 342, 400), np.float32)
    labelarteryall = np.zeros((40, 1, 342, 400), np.float32)
    labelveinall = np.zeros((40, 1, 342, 400), np.float32)
    labelvesselall = np.zeros((40, 1, 342, 400), np.float32)
    maskall = np.zeros((40, 1, 342, 400), np.float32)
    vessel = VesselProMap('./data/INSPIRE_AV/image')#, bgr2ggr=True, ggr_list=[i for i in range(1, 41)])#[4,7,10,11,19,23,26,28,32,33,35,36,37,39])
    rgg_list = [4,7,10,11,19,23,26,28,32,33,35,36,37,39]

    # load model
    n_classes = 3
    Net = TWGAN_Net(resnet='resnet18',input_ch=input_ch, num_classes= n_classes,use_cuda=use_cuda,pretrained=False, centerness=config.use_centerness, centerness_block_num=len(config.dilation_list))
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(36, 76)):
        rgb2rgg = True if (k-35) in rgg_list else False
        arterypred, veinpred, vesselpred, labelartery, labelvein, labelvessel, mask = GetResult(Net, k, vessel, use_cuda=use_cuda, rgb2rgg=rgb2rgg, config=config)

        arterypredall[k - 36, :, :, :] = cv2.resize(arterypred[0], (400, 342))
        veinpredall[k - 36, :, :, :] = cv2.resize(veinpred[0], (400, 342))
        vesselpredall[k - 36, :, :, :] = cv2.resize(vesselpred[0], (400, 342))
        labelarteryall[k - 36, :, :, :] = labelartery
        labelveinall[k - 36, :, :, :] = labelvein
        labelvesselall[k - 36, :, :, :] = labelvessel
        maskall[k - 36, :, :, :] = cv2.resize(mask[0].astype(np.float32), (400, 342))

    arteryauc, arteryacc, arterysp, arteryse, veinauc, veinacc, veinsp, veinse, bad_case_index = Evalution_AV_skeletonPixles(
        arterypredall, veinpredall, vesselpredall, labelarteryall, labelveinall, labelvesselall, maskall, 2, onlyMeasureSkeleton=True, strict_mode=strict_mode)
    # veinauc,veinacc,veinsp,veinse = evalution(veinpredall,labelveinall, maskall)
    vesselauc, vesselacc, vesselsp, vesselse = Evalution_HRF(vesselpredall, labelvesselall, maskall)
    promap2[:, 0, :, :] = arterypredall[:, 0, :, :]
    promap2[:, 1, :, :] = veinpredall[:, 0, :, :]
    promap2[:, 2, :, :] = vesselpredall[:, 0, :, :]
      
    labelmap[:, 0, :, :] = labelarteryall[:, 0, :, :]
    labelmap[:, 1, :, :] = labelveinall[:, 0, :, :]
    labelmap[:, 2, :, :] = labelvesselall[:, 0, :, :]

    filewriter = centerline_eval(promap2, config)
    #np.save(os.path.join(savePath, 'promap_inspire.npy'), promap2)
    np.save(os.path.join(inspire_path, "promap_inspire_testset.npy"), promap2)
    for k in range(0, 40):
        cv2.imwrite(os.path.join(inspire_path, "inspire_pro_" + str(k + 1) + ".png"), vessel[k, 0, :, :] * 255)
        cv2.imwrite(os.path.join(inspire_path, "inspire_artery_" + str(k + 1) + ".png"), arterypredall[k, 0, :, :] * 255)
        cv2.imwrite(os.path.join(inspire_path, "inspire_vein" + str(k + 1) + ".png"), veinpredall[k, 0, :, :] * 255)
        cv2.imwrite(os.path.join(inspire_path, "inspire_vessel" + str(k + 1) + ".png"), vesselpredall[k, 0, :, :] * 255)

    print("=============================inspire===========================")
    print("Strict mode:{}".format(strict_mode))
    print("the {} step arteryacc is:{}".format(i, arteryacc))
    print("the {} step arterysens is:{}".format(i, arteryse))
    print("the {} step arteryspec is:{}".format(i, arterysp))
    print("the {} step arteryauc is:{}".format(i, arteryauc))
    print("-----------------------------------------------------------")
    print("the {} step veinacc is:{}".format(i, veinacc))
    print("the {} step veinsens is:{}".format(i, veinse))
    print("the {} step veinspec is:{}".format(i, veinsp))
    print("the {} step veinauc is:{}".format(i, veinauc))
    print("-----------------------------------------------------------")
    print("the {} step vesselacc is:{}".format(i, vesselacc))
    print("the {} step vesselsens is:{}".format(i, vesselse))
    print("the {} step vesselspec is:{}".format(i, vesselsp))
    print("the {} step vesselauc is:{}".format(i, vesselauc))
    print("-----------------------------------------------------------")
    # print("the {} step loss is :{}".format(i,loss_all/200))
    if not os.path.exists(metrics_file_path):
        file_w = open(metrics_file_path, 'w')
    file_w = open(metrics_file_path, 'r+')
    file_w.read()
    file_w.write("=============================inspire===========================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 "the {} step arteryacc is:{}".format(i,arteryacc) + '\n' +
                 "the {} step arterysens is:{}".format(i,arteryse) + '\n' +
                 "the {} step arteryspec is:{}".format(i,arterysp) + '\n' +
                 "the {} step arteryauc is:{}".format(i,arteryauc) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "the {} step veinacc is:{}".format(i,veinacc) + '\n' +
                 "the {} step veinsens is:{}".format(i,veinse) + '\n' +
                 "the {} step veinspec is:{}".format(i,veinsp) + '\n' +
                 "the {} step veinauc is:{}".format(i,veinauc) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "the {} step vesselacc is:{}".format(i,vesselacc) + '\n'
                 "the {} step vesselsens is:{}".format(i,vesselse) + '\n'
                 "the {} step vesselspec is:{}".format(i,vesselsp) + '\n'
                 "the {} step vesselauc is:{}".format(i,vesselauc) + '\n'
                 "-----------------------------------------------------------" + '\n'
                 #"the {} step loss is :{}".format(i,loss_all/200) + '\n'
                 )
    file_w.write('\n')
    file_w.write(filewriter)
    file_w.close()


def modelEvalution_hrf(i,net,savePath,loss_all, use_cuda=False, dataset='hrf', is_kill_border=True, input_ch=3, strict_mode=False,config=None):
    # path for images to save
    hrf_path = os.path.join(savePath, 'hrf')
    metrics_file_path = os.path.join(savePath, 'metrics_'+str(config.k_fold)+'fold_'+str(config.k_fold_idx)+'.txt')
    print("metrics_file_path:",metrics_file_path)
    if not os.path.exists(hrf_path):
        os.mkdir(hrf_path)
    
    promap2 = np.zeros((15, 3, 800, 1200), np.float32)
    
    n_classes = 3
    Net = TWGAN_Net(resnet='resnet18',input_ch=input_ch, num_classes= n_classes,use_cuda=use_cuda,pretrained=False, centerness=config.use_centerness, centerness_block_num=len(config.dilation_list))
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    arterypredall = np.zeros((15, 1, 800, 1200), np.float32)
    veinpredall = np.zeros((15, 1, 800, 1200), np.float32)
    vesselpredall = np.zeros((15, 1, 800, 1200), np.float32)
    labelarteryall = np.zeros((15, 1, 800, 1200), np.float32)
    labelveinall = np.zeros((15, 1, 800, 1200), np.float32)
    labelvesselall = np.zeros((15, 1, 800, 1200), np.float32)
    maskall = np.zeros((15, 1, 800, 1200), np.float32)
    LabelMap = np.zeros((15, 3, 800, 1200), np.float32)
    vessel = VesselProMap('./data/HRF_AVLabel_191219/images/test')

    for k in tqdm(range(21,36)):
        arterypred,veinpred,vesselpred,labelartery,labelvein,labelvessel,mask = GetResult(Net,k,vessel, use_cuda=use_cuda,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config)
        arterypredall[k-21,:,:,:] = cv2.resize(arterypred[0],(1200,800))
        veinpredall[k-21,:,:,:] = cv2.resize(veinpred[0],(1200,800))
        vesselpredall[k-21,:,:,:] = cv2.resize(vesselpred[0],(1200,800))
        labelarteryall[k-21,:,:,:] = labelartery
        labelveinall[k-21,:,:,:] = labelvein
        labelvesselall[k-21,:,:,:] = labelvessel        
        maskall[k-21,:,:,:] = cv2.resize(mask[0].astype(np.float32),(1200,800))
    arteryauc,arteryacc,arterysp,arteryse,veinauc,veinacc,veinsp,veinse = Evalution_AV_skeletonPixles(arterypredall,veinpredall,vesselpredall,labelarteryall,labelveinall,labelvesselall,maskall,1,strict_mode=strict_mode)
    #veinauc,veinacc,veinsp,veinse = evalution(veinpredall,labelveinall, maskall)
    vesselauc,vesselacc,vesselsp,vesselse = Evalution_HRF(vesselpredall,labelvesselall, maskall)
    promap2[:,0,:,:] = arterypredall[:,0,:,:]
    promap2[:,1,:,:] = veinpredall[:,0,:,:]
    promap2[:,2,:,:] = vesselpredall[:,0,:,:]
    
    LabelMap[:,0,:,:] = labelarteryall[:,0,:,:]
    LabelMap[:,1,:,:] = labelveinall[:,0,:,:]
    LabelMap[:,2,:,:] = labelvesselall[:,0,:,:]
    
    filewriter = centerline_eval(promap2, config)

    np.save(os.path.join(hrf_path, "promap_hrf_testset_"+str(config.k_fold)+'fold_'+str(config.k_fold_idx)+".npy"), promap2)


    for k in range(0,15):
        cv2.imwrite(os.path.join(hrf_path, "hrf_pro_"+str(k+1)+".png"),vessel[k,0,:,:]*255)
        cv2.imwrite(os.path.join(hrf_path, "hrf_artery_"+str(k+1)+".png"),arterypredall[k,0,:,:]*255)
        cv2.imwrite(os.path.join(hrf_path, "hrf_vein"+str(k+1)+".png"),veinpredall[k,0,:,:]*255)
        cv2.imwrite(os.path.join(hrf_path, "hrf_vessel"+str(k+1)+".png"),vesselpredall[k,0,:,:]*255)
    
    print("=============================hrf===========================")
    print("Strict mode:{}".format(strict_mode))
    print("the {} step arteryacc is:{}".format(i,arteryacc))
    print("the {} step arterysens is:{}".format(i,arteryse))
    print("the {} step arteryspec is:{}".format(i,arterysp))
    print("the {} step arteryauc is:{}".format(i,arteryauc))
    print("-----------------------------------------------------------")
    print("the {} step veinacc is:{}".format(i,veinacc))
    print("the {} step veinsens is:{}".format(i,veinse))
    print("the {} step veinspec is:{}".format(i,veinsp))
    print("the {} step veinauc is:{}".format(i,veinauc))
    print("-----------------------------------------------------------")
    print("the {} step vesselacc is:{}".format(i,vesselacc))
    print("the {} step vesselsens is:{}".format(i,vesselse))
    print("the {} step vesselspec is:{}".format(i,vesselsp))
    print("the {} step vesselauc is:{}".format(i,vesselauc))
    print("-----------------------------------------------------------")
    #print("the {} step loss is :{}".format(i,loss_all/200))
    if not os.path.exists(metrics_file_path):
         file_w = open(metrics_file_path,'w')
    file_w = open(metrics_file_path,'r+')
    file_w.read()
    file_w.write("=============================hrf===========================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 "the {} step arteryacc is:{}".format(i,arteryacc) + '\n' +
                 "the {} step arterysens is:{}".format(i,arteryse) + '\n' +
                 "the {} step arteryspec is:{}".format(i,arterysp) + '\n' +
                 "the {} step arteryauc is:{}".format(i,arteryauc) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "the {} step veinacc is:{}".format(i,veinacc) + '\n' +
                 "the {} step veinsens is:{}".format(i,veinse) + '\n' +
                 "the {} step veinspec is:{}".format(i,veinsp) + '\n' +
                 "the {} step veinauc is:{}".format(i,veinauc) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "the {} step vesselacc is:{}".format(i,vesselacc) + '\n'
                 "the {} step vesselsens is:{}".format(i,vesselse) + '\n'
                 "the {} step vesselspec is:{}".format(i,vesselsp) + '\n'
                 "the {} step vesselauc is:{}".format(i,vesselauc) + '\n'
                 "-----------------------------------------------------------" + '\n'
                 #"the {} step loss is :{}".format(i,loss_all/200) + '\n'
                 ) 
    file_w.write(filewriter)
    file_w.close()


def modelEvalution(i,net,savePath,loss_all, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3, strict_mode=False,config=None):
    if dataset == 'hrf':
        modelEvalution_hrf(i,net,savePath,loss_all, use_cuda=use_cuda, dataset=dataset, is_kill_border=is_kill_border, input_ch=input_ch, strict_mode=strict_mode,config=config)
        return 
    # path for images to save
    drive_path = os.path.join(savePath, 'DRIVE')
    metrics_file_path = os.path.join(savePath, 'metrics.txt')#_'+str(config.model_step_pretrained_G)+'.txt')
    if not os.path.exists(drive_path):
        os.mkdir(drive_path)

    ArteryPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VeinPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VesselPredAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelArteryAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVeinAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVesselAll = np.zeros((20, 1, 584, 565), np.float32)
    ProMap = np.zeros((20, 3, 584, 565), np.float32)
    LabelMap = np.zeros((20, 3, 584, 565), np.float32)
    MaskAll = np.zeros((20, 1, 584, 565), np.float32)

    Vessel = VesselProMap('./data/AV_DRIVE/test/images')
    
        
    n_classes = 3
    Net = TWGAN_Net(resnet='resnet18', input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_block_num=len(config.dilation_list),centerness_map_size=config.centerness_map_size)
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(20)):
        ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask = GetResult(Net,k+1,Vessel,
                                                                                          use_cuda=use_cuda,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config)
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred
        LabelArteryAll[k,:,:,:] = LabelArtery
        LabelVeinAll[k,:,:,:] = LabelVein
        LabelVesselAll[k,:,:,:] = LabelVessel

        MaskAll[k,:,:,:] = Mask


    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]
    LabelMap[:,0,:,:] = LabelArteryAll[:,0,:,:]
    LabelMap[:,1,:,:] = LabelVeinAll[:,0,:,:]
    LabelMap[:,2,:,:] = LabelVesselAll[:,0,:,:]

    VesselAUC,VesselAcc,VesselSp,VesselSe = Evalution_drive(VesselPredAll, LabelVesselAll, MaskAll)

    filewriter = centerline_eval(ProMap, config)
    np.save(os.path.join(savePath, "ProMap_testset.npy"),ProMap)
    
    ArteryAUC,ArteryAcc,ArterySp,ArterySe,VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution_AV_skeletonPixles(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselAll,MaskAll,0,strict_mode=strict_mode)

    for k in range(0,20):
        cv2.imwrite(os.path.join(drive_path, "DRIVE_Pro_"+str(k+1)+".png"),Vessel[k,0,:,:]*255)
        cv2.imwrite(os.path.join(drive_path, "DRIVE_Artery_"+str(k+1)+".png"),ArteryPredAll[k,0,:,:]*255)
        cv2.imwrite(os.path.join(drive_path, "DRIVE_Vein"+str(k+1)+".png"),VeinPredAll[k,0,:,:]*255)
        cv2.imwrite(os.path.join(drive_path, "DRIVE_Vessel"+str(k+1)+".png"),VesselPredAll[k,0,:,:]*255)


    print("=========================DRIVE=============================")
    print("Strict mode:{}".format(strict_mode))
    print("The {} step ArteryAcc is:{}".format(i,ArteryAcc))
    print("The {} step ArterySens is:{}".format(i,ArterySe))
    print("The {} step ArterySpec is:{}".format(i,ArterySp))
    print("The {} step ArteryAUC is:{}".format(i,ArteryAUC))
    print("-----------------------------------------------------------")
    print("The {} step VeinAcc is:{}".format(i,VeinAcc))
    print("The {} step VeinSens is:{}".format(i,VeinSe))
    print("The {} step VeinSpec is:{}".format(i,VeinSp))
    print("The {} step VeinAUC is:{}".format(i,VeinAUC))
    print("-----------------------------------------------------------")
    print("The {} step VesselAcc is:{}".format(i,VesselAcc))
    print("The {} step VesselSens is:{}".format(i,VesselSe))
    print("The {} step VesselSpec is:{}".format(i,VesselSp))
    print("The {} step VesselAUC is:{}".format(i,VesselAUC))
    
    
    if not os.path.exists(metrics_file_path):
         file_w = open(metrics_file_path,'w')
    file_w = open(metrics_file_path,'r+')
    file_w.read()
    file_w.write("=========================DRIVE=============================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 "The {} step ArteryAcc is:{}".format(i,ArteryAcc) + '\n' +
                 "The {} step ArterySens is:{}".format(i,ArterySe) + '\n' +
                 "The {} step ArterySpec is:{}".format(i,ArterySp) + '\n' +
                 "The {} step ArteryAUC is:{}".format(i,ArteryAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "The {} step VeinAcc is:{}".format(i,VeinAcc) + '\n' +
                 "The {} step VeinSens is:{}".format(i,VeinSe) + '\n' +
                 "The {} step VeinSpec is:{}".format(i,VeinSp) + '\n' +
                 "The {} step VeinAUC is:{}".format(i,VeinAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "The {} step VesselAcc is:{}".format(i,VesselAcc) + '\n'
                 "The {} step VesselSens is:{}".format(i,VesselSe) + '\n'
                 "The {} step VesselSpec is:{}".format(i,VesselSp) + '\n'
                 "The {} step VesselAUC is:{}".format(i,VesselAUC) + '\n') 
    file_w.write(filewriter)
    file_w.close()



def GetResult(Net, k,Vessel, use_output_block= False, use_cuda=False, rgb2rgg=False, dataset='DRIVE', is_kill_border=True, input_ch=3, config=None):
    if k<=9:
        ImgName = './data/AV_DRIVE/test/images/0' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/0' + str(k) + '_test.png'
        Vessel0 = np.transpose(Vessel[k-1,:,:,:],(1,2,0))
        MaskName = './data/AV_DRIVE/test/mask/0' + str(k) + '_test_mask.png'
        Mask0 = cv2.imread(MaskName)
        Mask = np.zeros((Mask0.shape[0],Mask0.shape[1]),np.float32)
        Mask[Mask0[:,:,2]>0] = 1

    elif k<=20:
        ImgName = './data/AV_DRIVE/test/images/' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/' + str(k) + '_test.png'
        Vessel0 = np.transpose(Vessel[k-1,:,:,:],(1,2,0))
        MaskName = './data/AV_DRIVE/test/mask/' + str(k) + '_test_mask.png'
        Mask0 = cv2.imread(MaskName)
        Mask = np.zeros((Mask0.shape[0],Mask0.shape[1]),np.float32)
        Mask[Mask0[:,:,2]>0] = 1

    elif k<=35:
        ImgPath = './data/HRF_AVLabel_191219/images'#'./data/HRF_AVLabel/images/test'
        LabelPath = './data/HRF_AVLabel_191219/ArteryVein_0410_final'#'./data/HRF_AVLabel/label/test'
        ImgList0 , LabelList0 = getFolds(ImgPath, LabelPath, config.k_fold_idx, config.k_fold, trainset=False)
        ImgName = os.path.join(ImgPath, ImgList0[k-21])
        LabelName = os.path.join(LabelPath, LabelList0[k-21])
        Vessel0 = np.transpose(Vessel[k-21,:,:,:],(1,2,0))
    else: # 36-75
        ImgPath = './data/INSPIRE_AV/image'#'./data/HRF_AVLabel/images/test'
        ImgList0 = os.listdir(ImgPath)
        ImgList0 = natsort.natsorted(ImgList0)
        LabelPath = './data/INSPIRE_AV/label'#'./data/HRF_AVLabel/label/test'
        LabelList0 = os.listdir(LabelPath)
        LabelList0 = natsort.natsorted(LabelList0)
        ImgName = os.path.join(ImgPath, ImgList0[k-36])
        LabelName = os.path.join(LabelPath, LabelList0[k-36])
        Vessel0 = np.transpose(Vessel[k-36,:,:,:],(1,2,0))

    Img0 = cv2.imread(ImgName)
    Label0 = cv2.imread(LabelName)

    LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelArtery[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)] = 1
    LabelArtery[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVein[(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1
    LabelVein[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVessel[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1

    
    TempImg, TempMask = creatMask(Img0, threshold=10)

    IllumImage = illuminationCorrection(Img0, kernel_size=25, Mask=TempMask)

    if k>20:
        Mask = TempMask
    Img = Img0
    ImgIllCropped = IllumImage
    
    height, width = Img.shape[:2]
    
    n_classes = 3
    patch_height = 256
    patch_width = 256
    stride_height = 10
    stride_width = 10
    
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if rgb2rgg:
        Img[:,:,2:] = Img[:,:,1:2]
    Img = np.float32(Img/255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    Vessel_enlarged = paint_border_overlap_trad(Vessel0, patch_height, patch_width, stride_height, stride_width)

    ImgIllCropped = cv2.cvtColor(ImgIllCropped, cv2.COLOR_BGR2RGB)
    ImgIllCropped = np.float32(ImgIllCropped/255.)
    ImgIll_enlarged = paint_border_overlap(ImgIllCropped, patch_height, patch_width, stride_height, stride_width)


    patch_size = 256
    batch_size = 16
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    
    patches_vessel1 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,1)
    patches_vessel1 = np.transpose(patches_vessel1,(0,3,1,2))
    patches_vessel2 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,2)
    patches_vessel2 = np.transpose(patches_vessel2,(0,3,1,2))
    patches_vessel3 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,4)
    patches_vessel3 = np.transpose(patches_vessel3,(0,3,1,2))
    
    patches_imgsIll = extract_ordered_overlap(ImgIll_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgsIll = np.transpose(patches_imgsIll,(0,3,1,2))
    patches_imgsIll = Normalize(patches_imgsIll)

    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
               
        output_temp,_1 = Net(patches_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid
    
    
    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:,0:height,0:width]
    if is_kill_border:
        pred_img = kill_border(pred_img, Mask)
    
    ArteryPred = np.float32(pred_img[0,:,:])
    VeinPred = np.float32(pred_img[2,:,:])
    VesselPred = np.float32(pred_img[1,:,:])
    
    ArteryPred = ArteryPred[np.newaxis,:,:]
    VeinPred = VeinPred[np.newaxis,:,:]
    VesselPred = VesselPred[np.newaxis,:,:]
    LabelArtery = LabelArtery[np.newaxis,:,:]
    LabelVein = LabelVein[np.newaxis,:,:]
    LabelVessel = LabelVessel[np.newaxis,:,:]
    Mask = Mask[np.newaxis,:,:]
    
    
    return ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask


def Evalution_AV_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2, LabelVesselAll,MaskAll,DataSet=0,onlyMeasureSkeleton=False,strict_mode=False):
    threshold_confusion = 0.5
    y_scores1, y_true1,y_scores2, y_true2 = pred_only_FOV_AV(PredAll1,PredAll2,LabelAll1,LabelAll2, MaskAll,threshold_confusion)
    AUC1 = roc_auc_score(y_true1,y_scores1)   
    AUC2 = roc_auc_score(y_true2,y_scores2)
    if onlyMeasureSkeleton:
        accuracy1,specificity1,sensitivity1, bad_case_index = AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet,onlyMeasureSkeleton=onlyMeasureSkeleton,strict_mode=strict_mode)
        return AUC1, accuracy1, specificity1, sensitivity1, AUC2, accuracy1, sensitivity1, specificity1, bad_case_index
    else:
        accuracy1,specificity1,sensitivity1 = AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet,onlyMeasureSkeleton=onlyMeasureSkeleton,strict_mode=strict_mode)
    #accuracy2,specificity2,sensitivity2 = AVclassifiationMetrics(PredAll2,PredAll1,VesselPredAll,LabelAll2,LabelAll1,LabelVesselAll,DataSet)
    return AUC1,accuracy1,specificity1,sensitivity1,AUC2,accuracy1,sensitivity1,specificity1



def illuminationCorrection(Image, kernel_size, Mask):
    #input: original RGB image and kernel size
    #output: illumination corrected RGB image
    ## The return can be a RGB 3 channel image, but better to show the user the green channel only
    ##since green channel is more clear and has higher contrast

    Mask = np.uint8(Mask)
    Mask[Mask > 0] = 1
    Mask0 = Mask.copy()

    Img_pad = fakePad(Image, Mask, iterations=30)


    BackgroundIllumImage = cv2.medianBlur(Img_pad, ksize = kernel_size)

    maximumVal = np.max(BackgroundIllumImage)
    minimumVal = np.min(BackgroundIllumImage)
    constVal = maximumVal - 128

    BackgroundIllumImage[BackgroundIllumImage <=10] = 100
    IllumImage = Img_pad * (maximumVal / BackgroundIllumImage) - constVal
    IllumImage[IllumImage>255] = 255
    IllumImage[IllumImage<0] = 0
    IllumImage = np.uint8(IllumImage)

    IllumImage = cv2.bitwise_and(IllumImage, IllumImage, mask=Mask0)
    # IllumImage = cv2.medianBlur(IllumImage, ksize=3)

    return IllumImage

def draw_prediction(writer, pred, targs, step):
    target_artery = targs[0:1,0,:,:]
    target_vein   = targs[0:1,1,:,:]
    target_all    = targs[0:1,2,:,:]
    
    pred_sigmoid = pred #nn.Sigmoid()(pred)
    
    writer.add_image('artery',  torch.cat([pred_sigmoid[0:1,0,:,:], target_artery], dim=1), global_step=step)
    writer.add_image('vessel',  torch.cat([pred_sigmoid[0:1,1,:,:], target_all   ], dim=1), global_step=step)
    writer.add_image('vein',    torch.cat([pred_sigmoid[0:1,2,:,:], target_vein  ], dim=1), global_step=step)
    
