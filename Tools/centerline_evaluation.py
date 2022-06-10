import numpy as np
import cv2
import os
import natsort
from Tools.Hemelings_eval import evaluation_code
import pandas as pd

def getFolds(ImgPath, LabelPath, k_fold_idx,k_fold, trainset=True):
	for dirpath,dirnames,filenames in os.walk(ImgPath):
		ImgDirAll = filenames
		break

	for dirpath,dirnames,filenames in os.walk(LabelPath):
		LabelDirAll = filenames
		break
	ImgDir = []
	LabelDir = []
	
	ImgDir_testset = []
	LabelDir_testset = []

	if k_fold >0:
		ImgDirAll = natsort.natsorted(ImgDirAll)
		LabelDirAll = natsort.natsorted(LabelDirAll)
		num_fold = len(ImgDirAll) // k_fold
		for i in range(k_fold):
			start_idx = i * num_fold
			end_idx = (i+1) * num_fold
			if i == k_fold_idx:
				ImgDir_testset.extend(ImgDirAll[start_idx:end_idx])
				LabelDir_testset.extend(LabelDirAll[start_idx:end_idx])
				continue
			ImgDir.extend(ImgDirAll[start_idx:end_idx])
			LabelDir.extend(LabelDirAll[start_idx:end_idx])
	if not trainset:
		return ImgDir_testset, LabelDir_testset
	return ImgDir, LabelDir

def centerline_eval(ProMap, config):
	if config.dataset_name == 'hrf':
		ImgPath = os.path.join(config.trainset_path, 'images')
		LabelPath = os.path.join(config.trainset_path, 'ArteryVein_0410_final')
	elif config.dataset_name == 'DRIVE':
		dataroot = './data/AV_DRIVE/test'
		LabelPath = os.path.join(dataroot, 'av')
	else: #if config.dataset_name == 'INSPIRE':
		dataroot = './data/INSPIRE_AV'
		LabelPath = os.path.join(dataroot, 'label')
		DF_disc = pd.read_excel('./Tools/DiskParameters_INSPIRE_resize.xls', sheet_name=0)
	if config.dataset_name == 'hrf':
		k_fold_idx = config.k_fold_idx
		k_fold = config.k_fold

		ImgList0 , LabelList0 = getFolds(ImgPath, LabelPath, k_fold_idx,k_fold, trainset=False)
	overall_value = [[0,0] for i in range(3)]
	overall_value.append([0,0,0,0])
	overall_value.append(0)
	img_num = ProMap.shape[0]
	for i in range(img_num):
		arteryImg = ProMap[i, 0, :, :]
		veinImg   = ProMap[i, 1, :, :]
		vesselImg = ProMap[i, 2, :, :]

		idx = str(i+1)
		idx = idx.zfill(2)
		if config.dataset_name == 'hrf':
			imgName = ImgList0[i] 
		elif config.dataset_name == 'DRIVE':
			imgName = idx + '_test.png'
		else:
			imgName = 'image'+ str(i+1) + '_ManualAV.png'
		gt_path = os.path.join(LabelPath , imgName)
		print(gt_path)
		gt = cv2.imread(gt_path)
		gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
		if config.dataset_name == 'hrf':
			gt = cv2.resize(gt, (1200,800))
		gt_vessel = gt[:,:,0]+gt[:,:,2] 

		h, w = arteryImg.shape

		ArteryPred = np.float32(arteryImg)
		VeinPred   = np.float32(veinImg)
		VesselPred = np.float32(vesselImg)
		AVSeg1 = np.zeros((h, w, 3))
		vesselSeg = np.zeros((h, w))
		th = 0
		vesselPixels = np.where(gt_vessel)#(VesselPred>th) #

		for k in np.arange(len(vesselPixels[0])):
			row = vesselPixels[0][k]
			col = vesselPixels[1][k]
			if ArteryPred[row, col] >= VeinPred[row, col]:
				AVSeg1[row, col] = (255, 0, 0)
			else:
				AVSeg1[row, col] = ( 0, 0, 255)

		AVSeg1 = np.float32(AVSeg1)
		AVSeg1 = np.uint8(AVSeg1)

		if config.dataset_name == 'INSPIRE':
			discCenter = (DF_disc.loc[i, 'DiskCenterRow'], DF_disc.loc[i, 'DiskCenterCol'])
			discRadius = DF_disc.loc[i, 'DiskRadius']
			MaskDisc = np.ones((h, w), np.uint8)
			cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)
			out = evaluation_code(AVSeg1, gt, mask=MaskDisc,use_mask=True)
		else:
			out = evaluation_code(AVSeg1, gt)

		for j in range(len(out)):
			if j == 4:
				overall_value[j] += out[j]
				continue
			if j == 3:
				overall_value[j][0] += out[j][0]
				overall_value[j][1] += out[j][1]
				overall_value[j][2] += out[j][2]
				overall_value[j][3] += out[j][3]
				continue

			overall_value[j][0] += out[j][0]
			overall_value[j][1] += out[j][1]

	# print("overall_value:", overall_value)
	for j in range(len(overall_value)):
		if j == 4:
			overall_value[j] /= img_num
			continue
		if j == 3:
			overall_value[j][0] /= img_num
			overall_value[j][1] /= img_num
			overall_value[j][2] /= img_num
			overall_value[j][3] /= img_num
			continue
		overall_value[j][0] /= img_num
		overall_value[j][1] /= img_num
	# print
	metrics_names = ['full image', 'discovered centerline pixels', 'vessels wider than two pixels', 'all centerline', 'vessel detection rate']
	filewriter = ""
	print("--------------------------Centerline---------------------------------")
	filewriter += "--------------------------Centerline---------------------------------\n"
	for j in range(len(overall_value)):
		if j == 4:
			print("{} - Ratio:{}".format(metrics_names[j], overall_value[j]))
			filewriter += "{} - Ratio:{}\n".format(metrics_names[j], overall_value[j])
			continue
		if j == 3:
			print("{} - Acc: {} , F1:{}, Sens:{}, Spec:{}".format(metrics_names[j], overall_value[j][0],overall_value[j][1],overall_value[j][2],overall_value[j][3]))
			filewriter += "{} - Acc: {} , F1:{}, Sens:{}, Spec:{}\n".format(metrics_names[j], overall_value[j][0],overall_value[j][1],overall_value[j][2],overall_value[j][3])
			continue

		print("{} - Acc: {} , F1:{}".format(metrics_names[j], overall_value[j][0],overall_value[j][1]))
		filewriter += "{} - Acc: {} , F1:{}\n".format(metrics_names[j], overall_value[j][0],overall_value[j][1])
	
	return filewriter
