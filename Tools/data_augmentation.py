import numpy as np
import tensorlayer as tl

def data_augmentation1_9(image1, image4, image5, image6, image7, image8, image9, image10, image11):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = tl.prepro.rotation_multi([image1, image4, image5, image6, image7, image8, image9, image10, image11], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10, image11]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10, image11


def data_augmentation3_9(image1, image4, image5, image6, image7, image8, image9, image10, image11):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = tl.prepro.shift_multi([image1, image4, image5, image6, image7, image8, image9, image10, image11], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10, image11]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10, image11


def data_augmentation4_9(image1, image4, image5, image6, image7, image8, image9, image10, image11):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6, image7, image8, image9, image10, image11], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10, image11]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10, image11


def data_augmentation2_9(image1, image4, image5, image6, image7, image8, image9, image10, image11):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = tl.prepro.zoom_multi([image1, image4, image5, image6, image7, image8, image9, image10, image11], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10, image11] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10, image11]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10, image11

def data_augmentation1_8(image1, image4, image5, image6, image7, image8, image9, image10):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10] = tl.prepro.rotation_multi([image1, image4, image5, image6, image7, image8, image9, image10], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10


def data_augmentation3_8(image1, image4, image5, image6, image7, image8, image9, image10):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10] = tl.prepro.shift_multi([image1, image4, image5, image6, image7, image8, image9, image10], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10


def data_augmentation4_8(image1, image4, image5, image6, image7, image8, image9, image10):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6, image7, image8, image9, image10], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image6, image7, image8, image9, image10] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10


def data_augmentation2_8(image1, image4, image5, image6, image7, image8, image9, image10):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9, image10] = tl.prepro.zoom_multi([image1, image4, image5, image6, image7, image8, image9, image10], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9, image10] = np.squeeze([image1, image4, image5, image6, image7, image8, image9, image10]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9, image10

def data_augmentation1_7(image1, image4, image5, image6, image7, image8, image9):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9] = tl.prepro.rotation_multi([image1, image4, image5, image6, image7, image8, image9], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9] = np.squeeze([image1, image4, image5, image6, image7, image8, image9]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9


def data_augmentation3_7(image1, image4, image5, image6, image7, image8, image9):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9] = tl.prepro.shift_multi([image1, image4, image5, image6, image7, image8, image9], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9] = np.squeeze([image1, image4, image5, image6, image7, image8, image9]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9


def data_augmentation4_7(image1, image4, image5, image6, image7, image8, image9):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6, image7, image8, image9], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image6, image7, image8, image9] = np.squeeze([image1, image4, image5, image6, image7, image8, image9]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9


def data_augmentation2_7(image1, image4, image5, image6, image7, image8, image9):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8, image9] = tl.prepro.zoom_multi([image1, image4, image5, image6, image7, image8, image9], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6, image7, image8, image9] = np.squeeze([image1, image4, image5, image6, image7, image8, image9]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8, image9


def data_augmentation1_6(image1, image4, image5, image6, image7, image8):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8] = tl.prepro.rotation_multi([image1, image4, image5, image6, image7, image8], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6, image7, image8] = np.squeeze([image1, image4, image5, image6, image7, image8]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8


def data_augmentation3_6(image1, image4, image5, image6, image7, image8):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8] = tl.prepro.shift_multi([image1, image4, image5, image6, image7, image8], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6, image7, image8] = np.squeeze([image1, image4, image5, image6, image7, image8]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8


def data_augmentation4_6(image1, image4, image5, image6, image7, image8):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6, image7, image8], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image6, image7, image8] = np.squeeze([image1, image4, image5, image6, image7, image8]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8


def data_augmentation2_6(image1, image4, image5, image6, image7, image8):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7, image8] = tl.prepro.zoom_multi([image1, image4, image5, image6, image7, image8], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6, image7, image8] = np.squeeze([image1, image4, image5, image6, image7, image8]).astype(np.float32)

    return image1, image4, image5, image6, image7, image8

def data_augmentation1_5(image1, image4, image5, image6, image7):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7] = tl.prepro.rotation_multi([image1, image4, image5, image6, image7], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6, image7] = np.squeeze([image1, image4, image5, image6, image7]).astype(np.float32)

    return image1, image4, image5, image6, image7


def data_augmentation3_5(image1, image4, image5, image6, image7):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7] = tl.prepro.shift_multi([image1, image4, image5, image6, image7], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6, image7] = np.squeeze([image1, image4, image5, image6, image7]).astype(np.float32)

    return image1, image4, image5, image6, image7


def data_augmentation4_5(image1, image4, image5, image6, image7):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6, image7], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image, image7, image7] = np.squeeze([image1, image4, image5, image6, image7]).astype(np.float32)

    return image1, image4, image5, image6, image7


def data_augmentation2_5(image1, image4, image5, image6, image7):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6, image7] = tl.prepro.zoom_multi([image1, image4, image5, image6, image7], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6, image7] = np.squeeze([image1, image4, image5, image6, image7]).astype(np.float32)

    return image1, image4, image5, image6, image7

def data_augmentation1_4(image1, image4, image5, image6):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6] = tl.prepro.rotation_multi([image1, image4, image5, image6], rg=90, is_random=True,
                                                        fill_mode='constant')
    [image1, image4, image5, image6] = np.squeeze([image1, image4, image5, image6]).astype(np.float32)

    return image1, image4, image5, image6


def data_augmentation3_4(image1, image4, image5, image6):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6] = tl.prepro.shift_multi([image1, image4, image5, image6], wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    [image1, image4, image5, image6] = np.squeeze([image1, image4, image5, image6]).astype(np.float32)

    return image1, image4, image5, image6


def data_augmentation4_4(image1, image4, image5, image6):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6] = tl.prepro.elastic_transform_multi([image1, image4, image5, image6], alpha=720, sigma=24,
                                                                 is_random=True)
    [image1, image4, image5, image6] = np.squeeze([image1, image4, image5, image6]).astype(np.float32)

    return image1, image4, image5, image6


def data_augmentation2_4(image1, image4, image5, image6):
    # image3 = np.expand_dims(image3,-1)
    [image1, image4, image5, image6] = tl.prepro.zoom_multi([image1, image4, image5, image6], zoom_range=[0.7, 1.2], is_random=True,
                                                    fill_mode='constant')
    [image1, image4, image5, image6] = np.squeeze([image1, image4, image5, image6]).astype(np.float32)

    return image1, image4, image5, image6


def data_augmentation1_2(image1,image4,image5):
    #image3 = np.expand_dims(image3,-1)
    [image1,image4,image5] = tl.prepro.rotation_multi([image1,image4,image5] , rg=90, is_random=True, fill_mode='constant')
    [image1,image4,image5] = np.squeeze([image1,image4,image5]).astype(np.float32)
    
    return image1,image4,image5

def data_augmentation3_2(image1,image4,image5):
    #image3 = np.expand_dims(image3,-1)
    [image1,image4,image5] = tl.prepro.shift_multi([image1,image4,image5] ,  wrg=0.10,  hrg=0.10, is_random=True, fill_mode='constant')
    [image1,image4,image5] = np.squeeze([image1,image4,image5]).astype(np.float32)
    
    return image1,image4,image5

def data_augmentation4_2(image1,image4,image5):
    #image3 = np.expand_dims(image3,-1)
    [image1,image4,image5] = tl.prepro.elastic_transform_multi([image1,image4,image5], alpha=720, sigma=24, is_random=True)
    [image1,image4,image5] = np.squeeze([image1,image4,image5]).astype(np.float32)
    
    return image1,image4,image5

def data_augmentation2_2(image1,image4,image5):
    #image3 = np.expand_dims(image3,-1) 
    [image1,image4,image5] = tl.prepro.zoom_multi([image1,image4,image5] , zoom_range=[0.7, 1.2], is_random=True, fill_mode='constant')
    [image1,image4,image5] = np.squeeze([image1,image4,image5]).astype(np.float32)
    
    return image1,image4,image5

def data_augmentation1_2_h(image1):
    #image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.rotation_multi([image1] , rg=90, is_random=True, fill_mode='constant')   
    # [image1] = np.squeeze([image1]).astype(np.float32)
    image1 = np.squeeze([image1]).astype(np.float32)
    
    return image1

def data_augmentation3_2_h(image1):
    #image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.shift_multi([image1] ,  wrg=0.10,  hrg=0.10, is_random=True, fill_mode='constant')
    [image1] = np.squeeze([image1]).astype(np.float32)
    
    return image1

def data_augmentation4_2_h(image1):
    #image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.elastic_transform_multi([image1], alpha=720, sigma=24, is_random=True) 
    [image1] = np.squeeze([image1]).astype(np.float32)
    
    return image1

def data_augmentation2_2_h(image1):
    #image3 = np.expand_dims(image3,-1) 
    [image1] = tl.prepro.zoom_multi([image1] , zoom_range=[0.7, 1.2], is_random=True, fill_mode='constant')      
    # [image1] = np.squeeze([image1]).astype(np.float32) 
    image1 = np.squeeze([image1]).astype(np.float32) 
    return image1

###### data augmentation for shuffled dataset

def data_augmentation1_1(image1):
    # image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.rotation_multi([image1], rg=90, is_random=True, fill_mode='constant')
    image1 = np.squeeze([image1]).astype(np.float32)

    return image1


def data_augmentation3_1(image1):
    # image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.shift_multi([image1], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    image1 = np.squeeze([image1]).astype(np.float32)

    return image1


def data_augmentation4_1(image1):
    # image3 = np.expand_dims(image3,-1)
    [image1] = tl.prepro.elastic_transform_multi([image1], alpha=720, sigma=24, is_random=True)
    image1 = np.squeeze([image1]).astype(np.float32)

    return image1


def data_augmentation2_1(image1):
    # image3 = np.expand_dims(image3,-1) 
    [image1] = tl.prepro.zoom_multi([image1], zoom_range=[0.7, 1.2], is_random=True,
                                            fill_mode='constant')
    image1 = np.squeeze([image1]).astype(np.float32)

    return image1

def data_aug1(label_mat_fake, choice):
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    if choice == 0:
        label_mat_fake = label_mat_fake
    elif choice == 1:
        label_mat_fake = np.fliplr(label_mat_fake)
    elif choice == 2:
        label_mat_fake = np.flipud(label_mat_fake)
    elif choice == 3:
        label_mat_fake = data_augmentation1_1(label_mat_fake)
    elif choice == 4:
        label_mat_fake = data_augmentation2_1(label_mat_fake)
    elif choice == 5:
        label_mat_fake = data_augmentation3_1(label_mat_fake)
    elif choice == 6:
        label_mat_fake = data_augmentation4_1(label_mat_fake)

    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))

    return label_mat_fake


def data_aug4(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2 = data_augmentation1_4(data_mat_1, label_mat, label_mat_fake, label_mat_fake2)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2 = data_augmentation2_4(data_mat_1, label_mat, label_mat_fake, label_mat_fake2)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2 = data_augmentation3_4(data_mat_1, label_mat, label_mat_fake, label_mat_fake2)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2 = data_augmentation4_4(data_mat_1, label_mat, label_mat_fake, label_mat_fake2)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2

def data_aug5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    label_mat_fake3 = np.transpose(label_mat_fake3, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
        label_mat_fake3 = label_mat_fake3
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
        label_mat_fake3 = np.fliplr(label_mat_fake3)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
        label_mat_fake3 = np.flipud(label_mat_fake3)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3 = data_augmentation1_5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3 = data_augmentation2_5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3 = data_augmentation3_5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3 = data_augmentation4_5(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))
    label_mat_fake3 = np.transpose(label_mat_fake3, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3

def data_aug6(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    label_mat_fake3 = np.transpose(label_mat_fake3, (1, 2, 0))
    label_mat_fake4 = np.transpose(label_mat_fake4, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
        label_mat_fake3 = label_mat_fake3
        label_mat_fake4 = label_mat_fake4
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
        label_mat_fake3 = np.fliplr(label_mat_fake3)
        label_mat_fake4 = np.fliplr(label_mat_fake4)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
        label_mat_fake3 = np.flipud(label_mat_fake3)
        label_mat_fake4 = np.flipud(label_mat_fake4)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4 = data_augmentation1_6(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4 = data_augmentation2_6(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4 = data_augmentation3_6(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4 = data_augmentation4_6(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))
    label_mat_fake3 = np.transpose(label_mat_fake3, (2, 0, 1))
    label_mat_fake4 = np.transpose(label_mat_fake4, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4

def data_aug7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    label_mat_fake3 = np.transpose(label_mat_fake3, (1, 2, 0))
    label_mat_fake4 = np.transpose(label_mat_fake4, (1, 2, 0))
    label_mat_fake5 = np.transpose(label_mat_fake5, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
        label_mat_fake3 = label_mat_fake3
        label_mat_fake4 = label_mat_fake4
        label_mat_fake5 = label_mat_fake5
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
        label_mat_fake3 = np.fliplr(label_mat_fake3)
        label_mat_fake4 = np.fliplr(label_mat_fake4)
        label_mat_fake5 = np.fliplr(label_mat_fake5)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
        label_mat_fake3 = np.flipud(label_mat_fake3)
        label_mat_fake4 = np.flipud(label_mat_fake4)
        label_mat_fake5 = np.flipud(label_mat_fake5)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5 = data_augmentation1_7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5 = data_augmentation2_7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5 = data_augmentation3_7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5 = data_augmentation4_7(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))
    label_mat_fake3 = np.transpose(label_mat_fake3, (2, 0, 1))
    label_mat_fake4 = np.transpose(label_mat_fake4, (2, 0, 1))
    label_mat_fake5 = np.transpose(label_mat_fake5, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5


def data_aug8(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    label_mat_fake3 = np.transpose(label_mat_fake3, (1, 2, 0))
    label_mat_fake4 = np.transpose(label_mat_fake4, (1, 2, 0))
    label_mat_fake5 = np.transpose(label_mat_fake5, (1, 2, 0))
    label_mat_fake6 = np.transpose(label_mat_fake6, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
        label_mat_fake3 = label_mat_fake3
        label_mat_fake4 = label_mat_fake4
        label_mat_fake5 = label_mat_fake5
        label_mat_fake6 = label_mat_fake6
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
        label_mat_fake3 = np.fliplr(label_mat_fake3)
        label_mat_fake4 = np.fliplr(label_mat_fake4)
        label_mat_fake5 = np.fliplr(label_mat_fake5)
        label_mat_fake6 = np.fliplr(label_mat_fake6)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
        label_mat_fake3 = np.flipud(label_mat_fake3)
        label_mat_fake4 = np.flipud(label_mat_fake4)
        label_mat_fake5 = np.flipud(label_mat_fake5)
        label_mat_fake6 = np.flipud(label_mat_fake6)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6 = data_augmentation1_8(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6 = data_augmentation2_8(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6 = data_augmentation3_8(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6 = data_augmentation4_8(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))
    label_mat_fake3 = np.transpose(label_mat_fake3, (2, 0, 1))
    label_mat_fake4 = np.transpose(label_mat_fake4, (2, 0, 1))
    label_mat_fake5 = np.transpose(label_mat_fake5, (2, 0, 1))
    label_mat_fake6 = np.transpose(label_mat_fake6, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6

def data_aug9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7, choice):
    data_mat_1 = np.transpose(data_mat_1, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_mat_fake = np.transpose(label_mat_fake, (1, 2, 0))
    label_mat_fake2 = np.transpose(label_mat_fake2, (1, 2, 0))
    label_mat_fake3 = np.transpose(label_mat_fake3, (1, 2, 0))
    label_mat_fake4 = np.transpose(label_mat_fake4, (1, 2, 0))
    label_mat_fake5 = np.transpose(label_mat_fake5, (1, 2, 0))
    label_mat_fake6 = np.transpose(label_mat_fake6, (1, 2, 0))
    label_mat_fake7 = np.transpose(label_mat_fake7, (1, 2, 0))
    if choice == 0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
        label_mat_fake2 = label_mat_fake2
        label_mat_fake3 = label_mat_fake3
        label_mat_fake4 = label_mat_fake4
        label_mat_fake5 = label_mat_fake5
        label_mat_fake6 = label_mat_fake6
        label_mat_fake7 = label_mat_fake7
    elif choice == 1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
        label_mat_fake2 = np.fliplr(label_mat_fake2)
        label_mat_fake3 = np.fliplr(label_mat_fake3)
        label_mat_fake4 = np.fliplr(label_mat_fake4)
        label_mat_fake5 = np.fliplr(label_mat_fake5)
        label_mat_fake6 = np.fliplr(label_mat_fake6)
        label_mat_fake7 = np.fliplr(label_mat_fake7)
    elif choice == 2:
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
        label_mat_fake2 = np.flipud(label_mat_fake2)
        label_mat_fake3 = np.flipud(label_mat_fake3)
        label_mat_fake4 = np.flipud(label_mat_fake4)
        label_mat_fake5 = np.flipud(label_mat_fake5)
        label_mat_fake6 = np.flipud(label_mat_fake6)
        label_mat_fake7 = np.flipud(label_mat_fake7)
    elif choice == 3:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7 = data_augmentation1_9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7)
    elif choice == 4:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7 = data_augmentation2_9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7)
    elif choice == 5:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7 = data_augmentation3_9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7)
    elif choice == 6:
        data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7 = data_augmentation4_9(data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7)

    data_mat_1 = np.transpose(data_mat_1, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_mat_fake = np.transpose(label_mat_fake, (2, 0, 1))
    label_mat_fake2 = np.transpose(label_mat_fake2, (2, 0, 1))
    label_mat_fake3 = np.transpose(label_mat_fake3, (2, 0, 1))
    label_mat_fake4 = np.transpose(label_mat_fake4, (2, 0, 1))
    label_mat_fake5 = np.transpose(label_mat_fake5, (2, 0, 1))
    label_mat_fake6 = np.transpose(label_mat_fake6, (2, 0, 1))
    label_mat_fake7 = np.transpose(label_mat_fake7, (2, 0, 1))

    return data_mat_1, label_mat, label_mat_fake, label_mat_fake2, label_mat_fake3, label_mat_fake4, label_mat_fake5, label_mat_fake6, label_mat_fake7


def data_aug2(data_mat_1, label_mat, label_mat_fake, choice):
    data_mat_1 = np.transpose(data_mat_1,(1,2,0))
    label_mat = np.transpose(label_mat,(1,2,0))
    label_mat_fake = np.transpose(label_mat_fake,(1,2,0))
    if choice==0:
        data_mat_1 = data_mat_1
        label_mat = label_mat
        label_mat_fake = label_mat_fake
    elif choice==1:
        data_mat_1 = np.fliplr(data_mat_1)
        label_mat = np.fliplr(label_mat)
        label_mat_fake = np.fliplr(label_mat_fake)
    elif choice==2: 
        data_mat_1 = np.flipud(data_mat_1)
        label_mat = np.flipud(label_mat)
        label_mat_fake = np.flipud(label_mat_fake)
    elif choice==3:
        data_mat_1,label_mat, label_mat_fake = data_augmentation1_2(data_mat_1,label_mat, label_mat_fake)
    elif choice==4:
        data_mat_1,label_mat, label_mat_fake = data_augmentation2_2(data_mat_1,label_mat, label_mat_fake)
    elif choice==5:
        data_mat_1,label_mat, label_mat_fake = data_augmentation3_2(data_mat_1,label_mat, label_mat_fake)
    elif choice==6:
        data_mat_1,label_mat, label_mat_fake = data_augmentation4_2(data_mat_1,label_mat, label_mat_fake)
    
    data_mat_1 = np.transpose(data_mat_1,(2,0,1))
    label_mat = np.transpose(label_mat,(2,0,1))
    label_mat_fake = np.transpose(label_mat_fake,(2,0,1))
    
    return data_mat_1,label_mat,label_mat_fake
   
def data_aug (data_mat_1, choice):
    data_mat_1 = np.transpose(data_mat_1,(1,2,0))
    if choice==0:
        data_mat_1 = data_mat_1

    elif choice==1:
        data_mat_1 = np.fliplr(data_mat_1)

    elif choice==2: 
        data_mat_1 = np.flipud(data_mat_1)

    elif choice==3:
        data_mat_1 = data_augmentation1_2_h(data_mat_1)
    elif choice==4:
        data_mat_1 = data_augmentation2_2_h(data_mat_1)
    elif choice==5:
        data_mat_1 = data_augmentation3_2_h(data_mat_1)
    elif choice==6:
        data_mat_1 = data_augmentation4_2_h(data_mat_1)
    
    data_mat_1 = np.transpose(data_mat_1,(2,0,1))

    return data_mat_1
    