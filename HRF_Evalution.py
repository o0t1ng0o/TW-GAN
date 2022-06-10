# -*- coding: utf-8 -*-
###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import sys
sys.path.insert(0, './lib2/')
# help_functions.py
from help_functions2 import *
# extract_patches.py
from extract_patches2 import kill_border
from extract_patches2 import pred_only_FOV


def Evalution_HRF(preImg,masks_test, test_border_masks):
    #original test images (for FOV selection)
    full_img_height = preImg.shape[2]
    full_img_width = preImg.shape[3]
    

    pred_imgs = None
    #orig_imgs = None
    gtruth_masks = None
    pred_imgs = preImg
    gtruth_masks = masks_test  #ground truth masks
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
    ## back to original dimensions
    #orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    
    gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

    #====== Evaluate the results

    #predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV

    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration


    
    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)


    #Confusion matrix
    threshold_confusion = 0.5

    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)

    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))

    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])

    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])

    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
   
    #Jaccard similarity index
    #jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)

    
    #F1 score
    #F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

    
    return AUC_ROC,accuracy,specificity,sensitivity

