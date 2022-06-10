# valentina
import cv2
import numpy as np

def combineArteryVein(ASeg, VSeg):
    h,w = ASeg.shape
    AVSeg = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # white
            isRed = ASeg[i][j] == 0
            isBlue = VSeg[i][j] == 0
            if isRed and isBlue:
                AVSeg[i][j] = [0, 255, 0]
                continue
            if isRed:
                AVSeg[i][j] = [255, 0, 0]
            if isBlue:
                AVSeg[i][j] = [0, 0, 255]

    return AVSeg

def filterGreenWhite(AVSeg):
    for i in range(AVSeg.shape[0]):
        for j in range(AVSeg.shape[1]):
            if (AVSeg[i][j] == [0, 255, 0]).all():
                AVSeg[i][j] = [255, 0, 0]
            elif (AVSeg[i][j] == [255, 255, 255]).all():
                AVSeg[i][j] = [0, 0, 0]
    return AVSeg
def checkBlack(AVSeg, y, x, patch_h, patch_w):
    return sum(sum(sum(AVSeg[y:y+patch_h,x:x+patch_w,:]))) == 0
def getPatch(AVSeg, patch_h, patch_w):
    h, w, c = AVSeg.shape
    isBlack = True
    while isBlack:
        y = np.random.randint(0, h - patch_h)
        x = np.random.randint(0, w - patch_w)
        isBlack = checkBlack(AVSeg, y, x, patch_h, patch_w)
    return y, x

def removeArteryOrVein(patch):
    ArteryorVein = np.random.randint(0, 2) # [0, 1)
    # Artery
    if ArteryorVein == 0:
        # (255,0,0) -> (0,0,0)
        patch[:,:,0] = 0
    # Vein
    else:
        # (0,0,255) - > (0,0,0)
        patch[:,:,2] = 0
    return patch

def exchangeArteryVein(patch):
    # exchange artery and vein
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if (patch[i][j] == [255, 0, 0]).all():
                patch[i][j] = [0, 0, 255]
            elif (patch[i][j] == [0, 0, 255]).all():
                patch[i][j] = [255, 0, 0]
    return patch


def generate(AVSeg, shuffle_ratio = 0.1):
    img_h, img_w, _ = AVSeg.shape
    # get patch
    #rate = [0.02, 0.1] # [2%, 10%]

    newAVSeg = AVSeg.copy()


    totalPixNum = np.count_nonzero(AVSeg[:, :, 0] > 0) + np.count_nonzero(AVSeg[:, :, 2] > 0)

    changed_ratio = 0
    while changed_ratio < shuffle_ratio:
        size_rate = np.random.uniform(0.006, 0.04, 1)
        random_choice = np.random.randint(0, 3)

        patch_w = int(img_w * size_rate) #
        patch_h = int(img_h * size_rate) #

        y, x = getPatch(newAVSeg, patch_h, patch_w)
        input_patch = AVSeg[y:y+patch_h, x:x+patch_w, :].copy()

        # replace the artery or vein
        if random_choice == 0:
            patch = removeArteryOrVein(input_patch)
        ## exchange the result of different patch
        elif random_choice == 1:
            p_y, p_x = getPatch(AVSeg, patch_h, patch_w)
            patch = AVSeg[p_y:p_y+patch_h, p_x:p_x+patch_w, :].copy()
            #print("exchange_patch:", p_y, p_x)
        ## exchange the result of artery and vein in the same patch
        elif random_choice == 2:
            patch = exchangeArteryVein(input_patch)

        newAVSeg[y:y+patch_h, x:x+patch_w, :] = patch



        ##############Calculate Changed Ratio
        ChangedImg_Artery = AVSeg[:, :, 0] != newAVSeg[:, :, 0]
        ChangedImg_Artery = np.bitwise_and(AVSeg[:, :, 0] > 0, ChangedImg_Artery)
        ChangedImg_Vein = AVSeg[:, :, 2] != newAVSeg[:, :, 2]
        ChangedImg_Vein = np.bitwise_and(AVSeg[:, :, 2] > 0, ChangedImg_Vein)
        changeNum = np.count_nonzero(ChangedImg_Artery) + np.count_nonzero(ChangedImg_Vein)
        changed_ratio = changeNum/totalPixNum

    return newAVSeg