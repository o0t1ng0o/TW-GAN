from scipy import ndimage
import skimage.morphology as sm
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


print("Generating centerness maps for HRF dataset.")
print("The centerness maps are saved to : ./data/centerness_maps")

# setting for dilation
dil_list = [0,7,11]
dil_list_str = ''
for i in dil_list:
    dil_list_str += str(i)

print(dil_list)

ves_name_list = ["a", "v", "ves"]
max_value = 32.0

#--------------------- dilation = [0,7,11] ----------------------------
#[ 22.090722034374522,   22.80350850198276, 32.0] for training set (a,v,vessel)
#[26.870057685088806,  28.178005607210743，  28.178005607210743] for testset (a,v,vessel)
#----------------------------------------------------------------------
for ves_type in range(3):
    # 0 for artery
    # 1 for vein
    # 2 for vessel
    save_root = './data/centerness_maps/HRF_' + dil_list_str +'/'
    img_root = './data/HRF_AVLabel_191219/ArteryVein_0410_final/'
    
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    imglist = os.listdir(img_root)
    
    
    # make directory
    path_origin_dt = os.path.join(save_root, 'dilation_0')
    if not os.path.exists(path_origin_dt):
        os.mkdir(path_origin_dt)
    
    paths_list = []
    for i in dil_list[1:]: #range(start,end,gap):
        path = os.path.join(save_root, 'dilation_' + str(i))
        # print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        paths_list.append(path)
    
    max_dt = []
    
    for imgName in imglist:
        imgName_save     = ves_name_list[ves_type] + '_' + imgName
        imgName_save_dil = 'dil_' + ves_name_list[ves_type] + '_' + imgName
    
        imgName_save_cupless = 'dil_' + ves_name_list[ves_type] + '_' + imgName
    
        # img_save_path = os.path.join(save_root,imgName)
        Label0 = cv2.imread(os.path.join(img_root, imgName))
    
    
        LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
        LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
    
    
        LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
        LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
    
        Labels = [LabelArtery, LabelVein, LabelVessel]
    
    
        LabelArtery_dis = ndimage.distance_transform_edt(Labels[ves_type])
    
        mean_dis = LabelArtery_dis.mean()
        std_dis = LabelArtery_dis.std()
        max_dis = LabelArtery_dis.max()
        max_dt.append(max_dis)
        norm_dis = LabelArtery_dis / max_value  * 255#(LabelArtery_dis-mean_dis)/std_dis
    
    
        # save distance transform from original image
        cv2.imwrite(os.path.join(path_origin_dt, imgName_save), norm_dis)
        cv2.imwrite(os.path.join(path_origin_dt, imgName_save_cupless), Labels[ves_type]*255)
    
        j = 0
        for i in dil_list[1:]: #range(start,end,gap):
            LabelArtery_dil = sm.dilation(Labels[ves_type],sm.square(i))
            LabelArtery_dil_dis = ndimage.distance_transform_edt(LabelArtery_dil)
    
            mean_dil_dis = LabelArtery_dil_dis.mean()
            std_dil_dis = LabelArtery_dil_dis.std()
            max_dil_dis = LabelArtery_dil_dis.max()
            norm_dil_dis = LabelArtery_dil_dis/ max_value * 255#(LabelArtery_dil_dis - mean_dil_dis)/std_dil_dis
            max_dt.append(max_dil_dis)
    
            LabelArtery_dil = LabelArtery_dil * 255
            cv2.imwrite(os.path.join(paths_list[j], imgName_save_dil), LabelArtery_dil)
            cv2.imwrite(os.path.join(paths_list[j], imgName_save), norm_dil_dis)
            j +=1
    
    
    # print(max_dt)
    # print(str(ves_type)+"_max:", max(max_dt))

print("---------------------------------------")