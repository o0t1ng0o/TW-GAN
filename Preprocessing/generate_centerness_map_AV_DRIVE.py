from scipy import ndimage
import skimage.morphology as sm
import cv2
import numpy as np
import os
import pandas as pd
print("Generating centerness maps for AV-DRIVE dataset.")
print("The centerness maps are saved to : ./data/centerness_maps")

# setting for dilation
dil_list = [0,5,9]
dil_list_str = ''
for i in dil_list:
    dil_list_str += str(i)
print(dil_list)
height = 584
width = 565
ves_name_list = ["a", "v", "ves"]
max_value_list = [24.73863375370596, 28.160255680657446]
#--------------------- dilation = [0,5,9] ----------------------------
#[ 16.0312195418814, 18.867962264113206, 24.331050121192877] # for training set (a,v,vessel)
#[ 18.681541692269406, 20.248456731316587, 28.160255680657446] for test set (a,v,vessel)
#----------------------------------------------------------------------
dataset_list = ['training','test']
for dataset in dataset_list:
    print("Dataset:", dataset)
    max_value = max_value_list[1] if dataset == 'test' else max_value_list[0]
    filename = 'Test' if dataset =='test' else 'Train'
    DF_disc = pd.read_excel('./Tools/DiskParameters_DRIVE_' + filename + '.xls', sheet_name=0)
    if dataset == 'test':
        start_idx = 1
        end_idx =   21
    else:
        start_idx = 21
        end_idx = 41
    
    for ves_type in range(3):
        # 0 for artery
        # 1 for vein
        # 2 for vessel
        save_root = './data/centerness_maps/AV_DRIVE_' + dil_list_str + '/'+dataset+'/'
        img_root = './data/AV_DRIVE/'+ dataset +'/av/'
        
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
        
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
        idx = 0
        for img_idx in range(start_idx,end_idx,1):
            imgName = str(img_idx).zfill(2)+'_' + dataset + '.png'
            # print(imgName)
            imgName_save     = ves_name_list[ves_type] + '_' + imgName
            imgName_save_dil = 'dil_' + ves_name_list[ves_type] + '_' + imgName
            imgName_save_cupless = 'diskless_' + ves_name_list[ves_type] + '_' + imgName
        
            # img_save_path = os.path.join(save_root,imgName)
            Label0 = cv2.imread(img_root + imgName)
        
        
            LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
            LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
            LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.uint8)
        
        
            LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
            LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
            LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
            LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
            LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        
            Labels = [LabelArtery, LabelVein, LabelVessel]
        
            discCenter = (DF_disc.loc[idx, 'DiskCenterRow'], DF_disc.loc[idx, 'DiskCenterCol'])
            discRadius = DF_disc.loc[idx, 'DiskRadius']
            MaskDisc = np.ones((height, width), np.uint8)
            cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius=discRadius, color=0, thickness=-1)

            LabelArtery_cupless = LabelArtery * MaskDisc
            LabelVein_cupless   = LabelVein   * MaskDisc
            LabelVessel_cupless = LabelVessel * MaskDisc
            Labels_cupless = [LabelArtery_cupless, LabelVein_cupless, LabelVessel_cupless]
        
            LabelArtery_dis = ndimage.distance_transform_edt(Labels[ves_type])
        
            mean_dis = LabelArtery_dis.mean()
            std_dis = LabelArtery_dis.std()
            max_dis = LabelArtery_dis.max()
            max_dt.append(max_dis)
            norm_dis = LabelArtery_dis / max_value  * 255#(LabelArtery_dis-mean_dis)/std_dis
        
            # save distance transform from original image
            cv2.imwrite(os.path.join(path_origin_dt, imgName_save), norm_dis)
            cv2.imwrite(os.path.join(path_origin_dt, imgName_save_cupless), Labels_cupless[ves_type]*255)
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
            idx += 1
        
        # print(max_dt)
        # print(str(ves_type)+"_max:", max(max_dt))

print("---------------------------------------")