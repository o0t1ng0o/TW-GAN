import cv2
import numpy as np

def paint_border_overlap(img, patch_h, patch_w, stride_h, stride_w):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((img_h+(stride_h-leftover_h),img_w, 3))
        tmp_full_imgs[0:img_h,0:img_w, :] = img
        img = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((img.shape[0], img_w+(stride_w - leftover_w), 3))
        tmp_full_imgs[0:img.shape[0], 0:img_w, :] = img
        img = tmp_full_imgs
    return img

def paint_border_overlap_trad(img, patch_h, patch_w, stride_h, stride_w):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((img_h+(stride_h-leftover_h),img_w, 2))
        tmp_full_imgs[0:img_h,0:img_w, :] = img
        img = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((img.shape[0], img_w+(stride_w - leftover_w), 2))
        tmp_full_imgs[0:img.shape[0], 0:img_w, :] = img
        img = tmp_full_imgs
    return img

def pred_only_FOV_AV(data_imgs1,data_imgs2,data_masks1,data_masks2,original_imgs_border_masks,threshold_confusion):
    assert (len(data_imgs1.shape)==4 and len(data_masks1.shape)==4)  #4D arrays
    assert (data_imgs1.shape[0]==data_masks1.shape[0])
    assert (data_imgs1.shape[2]==data_masks1.shape[2])
    assert (data_imgs1.shape[3]==data_masks1.shape[3])
    assert (data_imgs1.shape[1]==1 and data_masks1.shape[1]==1)  #check the channel is 1
    height = data_imgs1.shape[2]
    width = data_imgs1.shape[3]
    new_pred_imgs1 = []
    new_pred_masks1 = []
    new_pred_imgs2 = []
    new_pred_masks2 = []
    for i in range(data_imgs1.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE_AV(i,x,y,data_masks1,data_masks2,original_imgs_border_masks,threshold_confusion)==True:
                    new_pred_imgs1.append(data_imgs1[i,:,y,x])
                    new_pred_masks1.append(data_masks1[i,:,y,x])
                    new_pred_imgs2.append(data_imgs2[i,:,y,x])
                    new_pred_masks2.append(data_masks2[i,:,y,x])
    new_pred_imgs1 = np.asarray(new_pred_imgs1)
    new_pred_masks1 = np.asarray(new_pred_masks1)
    new_pred_imgs2 = np.asarray(new_pred_imgs2)
    new_pred_masks2 = np.asarray(new_pred_masks2)
    return new_pred_imgs1, new_pred_masks1,new_pred_imgs2, new_pred_masks2

def inside_FOV_DRIVE_AV(i, x, y,data_imgs1,data_imgs2, DRIVE_masks,threshold_confusion):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[1]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0)&((data_imgs1[i,0,y,x]>threshold_confusion)|(data_imgs2[i,0,y,x]>threshold_confusion)):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False

def extract_ordered_overlap_trad(img, patch_h, patch_w,stride_h,stride_w,ratio):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    patches = np.empty((N_patches_img, patch_h//ratio, patch_w//ratio, 2))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            patch = img[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :]
            patch = cv2.resize(patch,(patch_h//ratio, patch_w//ratio))
            patches[iter_tot]=patch
            iter_tot +=1   #total
    assert (iter_tot==N_patches_img)
    return patches  #array with all the img divided in patches

def extract_ordered_overlap(img, patch_h, patch_w,stride_h,stride_w):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    patches = np.empty((N_patches_img, patch_h, patch_w, 3))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            patch = img[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :]
            patches[iter_tot]=patch
            iter_tot +=1   #total
    assert (iter_tot==N_patches_img)
    return patches  #array with all the img divided in patches


def pred_to_imgs(pred,mode="original"):
     assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
     assert (pred.shape[2]==2 )  #check the classes are 2
     pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
     if mode=="original":
         for i in range(pred.shape[0]):
             for pix in range(pred.shape[1]):
                 pred_images[i,pix]=pred[i,pix,1]
     elif mode=="threshold":
         for i in range(pred.shape[0]):
             for pix in range(pred.shape[1]):
                 if pred[i,pix,1]>=0.5:
                     pred_images[i,pix]=1
                 else:
                     pred_images[i,pix]=0
     else:
         print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
         exit()
     pred_images = np.reshape(pred_images,(pred_images.shape[0],1,48,48))
     return pred_images

def recompone_overlap(pred_patches, img_h, img_w, stride_h, stride_w):
    assert (len(pred_patches.shape)==4)  #4D arrays
    #assert (pred_patches.shape[1]==2 or pred_patches.shape[1]==3)  #check the channel is 1 or 3
    patch_h = pred_patches.shape[2]
    patch_w = pred_patches.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    #assert (pred_patches.shape[0]%N_patches_img==0)
    #N_full_imgs = pred_patches.shape[0]//N_patches_img
    full_prob = np.zeros((pred_patches.shape[1], img_h,img_w,))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((pred_patches.shape[1], img_h,img_w))

    k = 0 #iterator over all the patches
    for h in range(N_patches_h):
        for w in range(N_patches_w):
            full_prob[:, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]+=pred_patches[k]
            full_sum[:, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]+=1
            k+=1
    assert(k==pred_patches.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    #print(final_avg.shape)
    # assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    # assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


def Normalize(Patches):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    Patches[:,0,:,:] = (Patches[:,0,:,:] - mean[0]) / std[0]
    Patches[:,1,:,:] = (Patches[:,1,:,:] - mean[1]) / std[1]
    Patches[:,2,:,:] = (Patches[:,2,:,:] - mean[2]) / std[2]
    return Patches

def Normalize_patch(Patches):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    Patches[0,:,:] = (Patches[0,:,:] - mean[0]) / std[0]
    Patches[1,:,:] = (Patches[1,:,:] - mean[1]) / std[1]
    Patches[2,:,:] = (Patches[2,:,:] - mean[2]) / std[2]
    return Patches

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inside_FOV_DRIVE(x, y, DRIVE_masks):
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!
    if (x >= DRIVE_masks.shape[1] or y >= DRIVE_masks.shape[0]): #my image bigger than the original
        return False

    if (DRIVE_masks[y,x]>0):  #0==black pixels
        return True
    else:
        return False


def kill_border(pred_img, border_masks):
    height = pred_img.shape[1]
    width = pred_img.shape[2]
    for x in range(width):
        for y in range(height):
            if inside_FOV_DRIVE(x,y, border_masks)==False:
                pred_img[:,y,x]=0.0
    return pred_img

