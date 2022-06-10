import numpy as np
import cv2
import os
from tqdm import tqdm

root = 'D:/wenting/E/result/JCDA/DRIVE_trainset_gt_fake_0.01_0.05'
gt_root = 'D:/wenting/dataset/data/AV_DRIVE/training/av'
start = 21
end = 41

ratio_list = []
for i in range(start, end, 1):
    imgname_shuffled = os.path.join(root, str(i)+'_training.png')
    imgname_gt = os.path.join(gt_root, str(i)+'_training.png')
    img_gt = cv2.imread(imgname_gt) # BGR -> vein, ves, artery
    img_sf = cv2.imread(imgname_shuffled) # BGR

    # h, w, ch = img_gt.shape
    # totalPixNum = h*w*ch
    totalPixNum = np.count_nonzero(img_gt[:, :, 0] > 0) + np.count_nonzero(img_gt[:, :, 2] > 0)

    ##############Calculate Changed Ratio
    ChangedImg_Artery = img_gt[:, :, 0] != img_sf[:, :, 0]
    ChangedImg_Artery = np.bitwise_and(img_gt[:, :, 0] > 0, ChangedImg_Artery)
    ChangedImg_Vein = img_gt[:, :, 2] != img_sf[:, :, 2]
    ChangedImg_Vein = np.bitwise_and(img_gt[:, :, 2] > 0, ChangedImg_Vein)
    changeNum = np.count_nonzero(ChangedImg_Artery) + np.count_nonzero(ChangedImg_Vein)
    changed_ratio = changeNum / totalPixNum

    # changedImg = img_gt != img_sf
    # changeNum = np.count_nonzero(changedImg)
    # changed_ratio = changeNum / totalPixNum

    print(changed_ratio)
    ratio_list.append(changed_ratio)

sum_ratio = sum(ratio_list)
avg_ratio = sum_ratio / len(ratio_list)

print("sum:",sum_ratio)
print("avg:",avg_ratio)
print("max:",max(ratio_list))
print("min:",min(ratio_list))

