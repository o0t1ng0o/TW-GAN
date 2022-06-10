import torch
import os

# Check GPU availability
use_cuda = torch.cuda.is_available()
gpu_ids = [0] if use_cuda else []
device = torch.device('cuda' if use_cuda else 'cpu')

dataset_name = 'DRIVE'  # DRIVE or hrf 


max_step = 30000  # 30000 for DRIVE
patch_size_list = [96, 128, 256]  
patch_size = patch_size_list[2]
batch_size = 4 # default: 4
print_iter = 100 # default: 100
display_iter = 100 # default: 100
save_iter = 5000 # default: 5000
lr = 0.0002 # default: 0.0002
step_size = 7000  # 7000 for DRIVE
lr_decay_gamma = 0.5  # default: 0.5
use_SGD = False # default:False

input_nc = 3
ndf = 32
netD_type = 'basic'
n_layers_D = 5
norm = 'instance'
no_lsgan = False
init_type = 'normal'
init_gain = 0.02
use_sigmoid = no_lsgan

# torch.cuda.set_device(gpu_ids[0])
use_GAN = True # default: True

GAN_type = 'rank'  # 'vanilla' ,'wgan', 'rank'
treat_fake_cls0 = False  
use_noise_input_D = False  # whether use the noisy image as the input of discriminator
use_dropout_D = False  # whether use dropout in each layer of discriminator
vgg_type = 'vgg19'
vgg_layers = [4, 9, 18, 27]
lambda_vgg = 1


# settings for triplet loss
use_triplet_loss = True  # whether use triplet loss
margin = 1  # default : 1
triplet_anchor = 'real'  # default : real
lambda_triplet_list = [1,1,1] # A,V,Vessel
lambda_triplet = 0.1
beta1 = 0.5
num_classes_D = 3
lambda_GAN_D = 0.01
lambda_GAN_G = 0.01
lambda_GAN_gp = 100
lambda_BCE = 5
lambda_recon = 0
overlap_vessel = 0  # default: 0 (both artery and vein); 1 (artery) ; 2 (vein)

input_nc_D = input_nc + 3

# settings for centerness
use_centerness =True # default: True
dilation_list =  [0,5,9] if dataset_name == 'DRIVE' else [0,7,11] # default: [0,5,9], [0,7,11] for HRF
lambda_centerness = 1
lambda_dilation_list = [1,1,1] # default :[1,1,1]
lambda_dilation_list = [ w*lambda_centerness for w in lambda_dilation_list]
center_loss_type = 'centerness' # centerness or smoothl1
dilation_list_str = '_'
for i in dilation_list:
    dilation_list_str += str(i)
assert len(lambda_dilation_list) == len(dilation_list)
centerness_map_size =  [128,128]

# pretrained model
use_pretrained_G = False 
model_path_pretrained_G = ''
model_step_pretrained_G = 0

# k fold cross validation for HRF dataset
k_fold_idx = 2
k_fold = 3

# path for dataset

n_classes = 3

model_step = 0


dataset_path = {'DRIVE': './data/AV_DRIVE/training/',
                'DRIVE_centerness': './data/centerness_maps/AV_DRIVE' + dilation_list_str + '/training/',
                'DRIVE_shuffle': './data/shuffled_dataset/AV_DRIVE/training/',
                'hrf': './data/HRF_AVLabel_191219/',
                'hrf_centerness': './data/centerness_maps/HRF' + dilation_list_str + '/',
                'hrf_shuffle': './data/shuffled_dataset/HRF/'
                }
trainset_path = dataset_path[dataset_name]
trainset_shuffle_path = dataset_path[dataset_name + '_shuffle']
trainset_centerness_path = dataset_path[dataset_name + '_centerness']
print("Dataset:")
print(trainset_path)
print(trainset_shuffle_path)
print(trainset_centerness_path)



