import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from models.network import TWGAN_Net, set_requires_grad, VGGNet

from models import networks_gan
from loss import multiclassLoss, CrossEntropyLossWithSmooth, L1LossWithLogits, vggloss, gradient_penalty, tripletMarginLoss_vggfea, centernessLoss, SmoothL1_weighted
import os, copy
from opt import get_patch_trad_5, get_patch_trad_7, get_patch_trad_9, modelEvalution
from collections import OrderedDict
import numpy as np
import cv2
from Preprocessing.autoaugment import ImageNetPolicy
class TW_GAN():

    def __init__(self, opt, isTrain=True):
        self.cfg = opt
        self.use_GAN = opt.use_GAN
        self.isTrain = isTrain
        self.use_cuda = opt.use_cuda
        self.centerness_block_num = len(opt.dilation_list)
        # initilize all the loss names for print in each iteration
        self.get_loss_names(opt)
        self.centerness_map_size = opt.centerness_map_size

        # define networks (both generator and discriminator)
        self.netG = TWGAN_Net(resnet = 'resnet18', pretrained=True,
                             input_ch=opt.input_nc,
                             num_classes = opt.n_classes,
                             centerness=opt.use_centerness, 
                             centerness_block_num=self.centerness_block_num,
                             centerness_map_size=opt.centerness_map_size)
        if self.use_cuda:
            self.netG = self.netG.cuda()


        if self.isTrain and opt.use_GAN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            tmp_num_classes_D = opt.num_classes_D
            if opt.GAN_type == 'rank':
                tmp_num_classes_D = 2
            self.netD = networks_gan.define_D(input_nc=opt.input_nc_D, ndf=opt.ndf,
                                              netD=opt.netD_type, n_layers_D=opt.n_layers_D,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid,
                                              init_type=opt.init_type, init_gain=opt.init_gain,
                                              gpu_ids=opt.gpu_ids,num_classes_D=tmp_num_classes_D,
                                              use_noise=opt.use_noise_input_D, use_dropout=opt.use_dropout_D)
            print(self.netD)
            self.netD.train()
            self.netG.train()



        if self.isTrain:

            # define loss functions
            if opt.GAN_type == 'vanilla':
                self.criterionCE = CrossEntropyLossWithSmooth(opt.label_smoothing, opt.num_classes_D,use_cuda=opt.use_cuda) \
                                    if opt.use_label_smooth \
                                    else nn.CrossEntropyLoss()
            elif opt.GAN_type == 'rank':
                self.criterionCE = nn.BCEWithLogitsLoss()

            self.criterion = multiclassLoss()

            # initialize optimizers and scheduler.
            if opt.use_SGD:
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr,momentum = 0.9,weight_decay=5e-4)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G,step_size =opt.step_size, gamma=0.5)
            if opt.use_GAN:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opt.step_size, gamma=opt.lr_decay_gamma)

            # pre-define feature extractor
            # define VGG
            if opt.use_triplet_loss :
                self.vggnet = VGGNet(opt.vgg_type, opt.vgg_layers, opt.use_cuda)
                self.vggnet.vgg.to(opt.device)

            # define loss function and feature extractor for triplet loss module
            if opt.use_triplet_loss:
                self.criterionTriplet = nn.TripletMarginLoss(p=2, margin=opt.margin)

            # define loss function for centerness score map
            if opt.use_centerness:
                self.criterionSmoothL1 = nn.SmoothL1Loss(reduction='none')
                self.criterionSmoothL1_mean = SmoothL1_weighted(weight_list=opt.lambda_dilation_list)


    def setup(self, opt, log_folder):
        # define the directory for logger
        self.log_folder = log_folder
        # mkdir for training result
        self.train_result_folder = os.path.join(self.log_folder, 'training_result')
        if not os.path.exists(self.train_result_folder):
            os.mkdir(self.train_result_folder)

        # load network
        if not self.isTrain or opt.use_pretrained_G:
            model_path = os.path.join(opt.model_path_pretrained_G, 'G_' + str(opt.model_step_pretrained_G) + '.pkl')
            self.netG.load_state_dict(torch.load(model_path))
            print("Loading pretrained model for Generator from " + model_path)
            if opt.use_GAN:
                model_path_D = os.path.join(opt.model_path_pretrained_G, 'D_' + str(opt.model_step_pretrained_G) + '.pkl')
                self.netD.load_state_dict(torch.load(model_path_D))
                print("Loading pretrained model for Discriminator from " + model_path_D)

    def set_input(self, step, train_data1=None, label_data=None, label_data_fake=None, label_data_centerness=None, label_data_dilation=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """

        opt = self.cfg

        self.step = step

        num_map = len(label_data_centerness)
        if num_map == 1 :
            data1, label, label_fake, label_centerness, label_nodisk = get_patch_trad_5(opt.batch_size, opt.patch_size, train_data1, label_data,
                                                        label_data_fake, label_data_centerness[0], label_data_dilation[0],
                                                        patchsize1=opt.patch_size_list[0], patchsize2=opt.patch_size_list[1])
        elif num_map == 2 :
            data1, label, label_fake, label_centerness, label_centerness2, label_nodisk, label_dilation2 = \
                                                        get_patch_trad_7(opt.batch_size, opt.patch_size, train_data1, label_data,
                                                        label_data_fake, label_data_centerness[0], label_data_centerness[1], label_data_dilation[0], label_data_dilation[1],
                                                        patchsize1=opt.patch_size_list[0], patchsize2=opt.patch_size_list[1])
        elif num_map == 3 :
            data1, label, label_fake, label_centerness, label_centerness2, label_centerness3, label_nodisk, label_dilation2, label_dilation3 = \
                                                        get_patch_trad_9(opt.batch_size, opt.patch_size, train_data1, label_data,
                                                        label_data_fake, label_data_centerness[0], label_data_centerness[1], label_data_centerness[2], 
                                                        label_data_dilation[0],label_data_dilation[1],label_data_dilation[2], 
                                                        patchsize1=opt.patch_size_list[0], patchsize2=opt.patch_size_list[1])#, policy=self.autoaugment_policy)
                                                        

        self.data_input1 = torch.FloatTensor(data1)
        self.label_input = torch.FloatTensor(label)

        self.label_centerness_map = torch.FloatTensor(label_centerness)
        self.label_centerness_map2 = torch.FloatTensor(label_centerness2) if num_map >=2 else None 
        self.label_centerness_map3 = torch.FloatTensor(label_centerness3) if num_map ==3 else None
        self.label_input_sm = torch.FloatTensor(copy.deepcopy(label))
        
        self.label_nodisk_map = torch.FloatTensor(label_nodisk)
        self.label_dilation_map2 = torch.FloatTensor(label_dilation2) if num_map >=2 else None 
        self.label_dilation_map3 = torch.FloatTensor(label_dilation3) if num_map ==3 else None


        if opt.use_cuda:
            self.data_input1 = self.data_input1.cuda()
            self.label_input = self.label_input.cuda()
            self.label_centerness_map = self.label_centerness_map.cuda()
            self.label_centerness_map2 = self.label_centerness_map2.cuda() if num_map >=2 else None 
            self.label_centerness_map3 = self.label_centerness_map3.cuda() if num_map ==3 else None
            self.label_input_sm = self.label_input_sm.cuda()

            self.label_nodisk_map = self.label_nodisk_map.cuda()
            self.label_dilation_map2 = self.label_dilation_map2.cuda() if num_map >=2 else None 
            self.label_dilation_map3 = self.label_dilation_map3.cuda() if num_map ==3 else None

        # downsample the centerness scores maps and dilated images
        if self.centerness_map_size[0] == 128:
            self.label_centerness_map = F.interpolate(self.label_centerness_map, scale_factor=0.5, mode='bilinear', align_corners=True)
            self.label_centerness_map2 = F.interpolate(self.label_centerness_map2, scale_factor=0.5, mode='bilinear', align_corners=True) if num_map >=2 else None
            self.label_centerness_map3 = F.interpolate(self.label_centerness_map3, scale_factor=0.5, mode='bilinear', align_corners=True) if num_map ==3 else None
            self.label_input_sm = F.interpolate(self.label_input_sm, scale_factor=0.5, mode='bilinear', align_corners=True) 
            self.label_nodisk_map = F.interpolate(self.label_nodisk_map, scale_factor=0.5, mode='bilinear', align_corners=True) 
            self.label_dilation_map2 = F.interpolate(self.label_dilation_map2, scale_factor=0.5, mode='bilinear', align_corners=True) if num_map >=2 else None
            self.label_dilation_map3 = F.interpolate(self.label_dilation_map3, scale_factor=0.5, mode='bilinear', align_corners=True) if num_map ==3 else None

        self.data_input1 = autograd.Variable(self.data_input1)
        self.label_input = autograd.Variable(self.label_input)

        self.label_centerness_map = autograd.Variable(self.label_centerness_map)
        self.label_centerness_map2 = autograd.Variable(self.label_centerness_map2) if num_map >=2 else None 
        self.label_centerness_map3 = autograd.Variable(self.label_centerness_map3) if num_map ==3 else None
        self.label_input_sm = autograd.Variable(self.label_input_sm)
        self.label_nodisk_map = autograd.Variable(self.label_nodisk_map)
        self.label_dilation_map2 = autograd.Variable(self.label_dilation_map2) if num_map >=2 else None 
        self.label_dilation_map3 = autograd.Variable(self.label_dilation_map3) if num_map ==3 else None

        # concat all the centerness score map label
        self.label_centerness_map_all = [self.label_centerness_map]
        if num_map >= 2:
            self.label_centerness_map_all.append(self.label_centerness_map2)
        if num_map >= 3:
            self.label_centerness_map_all.append(self.label_centerness_map3)
        self.label_centerness_map_all = torch.cat(self.label_centerness_map_all, dim=1)

        self.label_fake_input = torch.FloatTensor(label_fake)
        if opt.use_cuda:
            self.label_fake_input = self.label_fake_input.cuda()
        self.label_fake_input = autograd.Variable(self.label_fake_input)

        self.input = self.data_input1


    def forward(self):

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.pre_target, self.centerness_maps = self.netG(self.input)
        # sigmoid
        self.pre_target = torch.sigmoid(self.pre_target)

    def save_model(self):
        # save generator
        torch.save(self.netG.state_dict(), os.path.join(self.log_folder, 'G_' + str(self.step)+'.pkl'))
        torch.save(self.netG, os.path.join(self.log_folder, 'G_' + str(self.step)+'.pth'))
        # save discriminator
        if self.cfg.use_GAN:
            torch.save(self.netD.state_dict(), os.path.join(self.log_folder, 'D_' + str(self.step)+'.pkl'))
            torch.save(self.netD, os.path.join(self.log_folder, 'D_' + str(self.step)+'.pth'))
        print("save model to {}".format(self.log_folder))

    def log(self, logger):
        logger.draw_prediction(self.pre_target, self.label_input, self.centerness_maps, self.label_centerness_map_all, self.step)

    def get_loss_names(self, opt):
        self.loss_names = []
        if opt.use_GAN:
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')
            if opt.GAN_type == 'vanilla' and opt.num_classes_D == 3 or opt.GAN_type == 'rank':
                self.loss_names.append('D_real_shuffle')
            if opt.GAN_type == 'wgan':
                self.loss_names.append('D_gp')
            self.loss_names.append('D')
            self.loss_names.append('G_GAN')

        self.loss_names.append('G_BCE')
        if opt.use_triplet_loss:
            self.loss_names.append('G_triplet')
        self.loss_names.append('G')
        if opt.use_centerness:
            self.loss_names.append('G_centerness')

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_ret

    def test(self, result_folder):
        print("-----------start to test-----------")
        modelEvalution(self.step, self.netG.state_dict(),
                       result_folder, 0,
                       use_cuda=self.cfg.use_cuda,
                       dataset=self.cfg.dataset_name,
                       input_ch=self.cfg.input_nc,
                       config=self.cfg,
                       strict_mode=True)
        print("---------------end-----------------")

    def backward_D(self, isBackward=True):

        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        opt = self.cfg

        # define the input of D
        real_input = torch.cat([self.data_input1, self.label_input], dim=1)
        fake_input = torch.cat([self.data_input1, self.pre_target], dim=1)
        real_shuffle_input = torch.cat([self.data_input1, self.label_fake_input], dim=1)

        pred_real = self.netD(real_input)
        pred_fake = self.netD(fake_input.detach())  # bs x ch x (HxW)
        pred_real_shuffle = self.netD(real_shuffle_input)  # shuffle(label_input))
        # pred_fake_shuffle = netD(shuffle(pre_target.detach()))

        # Compute loss
        self.loss_D = 0

        # for GT
        label_shape = [opt.batch_size, pred_real.shape[2]]
        # 0, 1, 2
        label_real = torch.zeros(label_shape).long()
        label_fake = torch.ones(label_shape).long()
        label_real_sf = torch.ones(label_shape).long() * 2
        if opt.use_cuda:
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
            label_real_sf = label_real_sf.cuda()
        if opt.GAN_type == 'vanilla':
            self.loss_D_real = self.criterionCE(pred_real, label_real)
            self.loss_D_fake = self.criterionCE(pred_fake, label_fake)
            self.loss_D_real_shuffle = self.criterionCE(pred_real_shuffle, label_real_sf) if opt.num_classes_D == 3 else 0
            self.loss_D = (self.loss_D_real + self.loss_D_fake + self.loss_D_real_shuffle)
            self.loss_D = self.loss_D * opt.lambda_GAN_D  # loss_D_fake_shuffle

        elif opt.GAN_type == 'wgan':  # wgan
            self.loss_D_real = - torch.mean(pred_real)
            self.loss_D_fake = torch.mean(pred_fake)

            # Compute loss for gradient penalty.
            alpha = torch.rand(real_input.size(0), 1, 1, 1).to(opt.device)
            real_hat = (alpha * real_input.data + (1 - alpha) * fake_input.data).requires_grad_(True)
            pred_real_hat = self.netD(real_hat)
            self.loss_D_gp = gradient_penalty(pred_real_hat, real_hat, opt.device)
            self.loss_D = opt.lambda_GAN_D * (self.loss_D_real + self.loss_D_fake) + opt.lambda_GAN_gp * self.loss_D_gp
            # loss_D_fake_shuffle = criterionCE(pred_fake_shuffle, torch.ones([batch_size, pred_real.shape[2]]).long() * 3)
        elif opt.GAN_type == 'rank':  # rank
            # fake          [0, 0 ]
            # real_shuffle  [1, 0 ]
            # real          [1, 1 ]
            multi_label_shape = [opt.batch_size, 2, label_shape[1]]  # bs x 2 x (hxw)
            class_0 = torch.zeros(multi_label_shape)  # 0 0
            class_1 = torch.ones(multi_label_shape)  # 1 0
            class_1[:, 1, :] = 0
            # class_1 = class_1.long()
            class_2 = torch.ones(multi_label_shape)  # 1 1
            if opt.use_cuda:
                class_0 = class_0.cuda()
                class_1 = class_1.cuda()
                class_2 = class_2.cuda()

            loss_D_fake = self.criterionCE(pred_fake, class_0) if opt.treat_fake_cls0 else self.criterionCE(pred_fake, class_1)
            loss_D_real_shuffle = self.criterionCE(pred_real_shuffle, class_1) if opt.treat_fake_cls0 else self.criterionCE(
                pred_real_shuffle, class_0)
            loss_D_real = self.criterionCE(pred_real, class_2)

            self.loss_D_fake = loss_D_fake * opt.lambda_GAN_D
            self.loss_D_real_shuffle = loss_D_real_shuffle * opt.lambda_GAN_D
            self.loss_D_real = loss_D_real * opt.lambda_GAN_D

            self.loss_D += loss_D_real + loss_D_fake + loss_D_real_shuffle

        # backward
        if isBackward:
            self.loss_D.backward()
    def backward_G(self, isBackward=True):

        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        opt = self.cfg

        # define input
        fake_input_cpy = torch.cat([self.data_input1, self.pre_target], dim=1)
        self.loss_G = 0

        # GAN
        if opt.use_GAN:
            pred_fake = self.netD(fake_input_cpy)
            if opt.GAN_type == 'vanilla':
                zeros_tensor = torch.zeros([opt.batch_size, self.pred_real.shape[2]]).long()
                if opt.use_cuda:
                    zeros_tensor = zeros_tensor.cuda()
                self.loss_G_GAN += opt.lambda_GAN_G * self.criterionCE(pred_fake, zeros_tensor)
            elif opt.GAN_type == 'wgan':
                self.loss_G_GAN += opt.lambda_GAN_G * (- torch.mean(pred_fake))
            elif opt.GAN_type == 'rank':
                multi_label_shape = [opt.batch_size, 2, pred_fake.shape[2]]  # bs x 2 x (hxw)
                class_2 = torch.ones(multi_label_shape)  # 1 1
                if opt.use_cuda:
                    class_2 = class_2.cuda()
                self.loss_G_GAN = opt.lambda_GAN_G * self.criterionCE(pred_fake, class_2)
            self.loss_G += self.loss_G_GAN

        # BCE
        self.loss_G_BCE = opt.lambda_BCE * self.criterion(self.pre_target, self.label_input)
        self.loss_G += self.loss_G_BCE

        # Triplet loss
        if opt.use_triplet_loss:
            # use VGG19 as feature extractor
            loss_G_triplet, dis1, dis2 = tripletMarginLoss_vggfea(self.vggnet, self.criterionTriplet,
                                                                         self.pre_target, self.label_input,
                                                                         self.label_fake_input,
                                                                         use_cuda=opt.use_cuda,
                                                                         anchor=opt.triplet_anchor,
                                                                         weight_list=opt.lambda_triplet_list)
            # display the distance between positive and negative pairs
            self.loss_G_triplet_a_pos = dis1[0].item()
            self.loss_G_triplet_a_neg = dis2[0].item()
            self.loss_G_triplet_v_pos = dis1[1].item()
            self.loss_G_triplet_v_neg = dis2[1].item()
            self.loss_G_triplet_ves_pos = dis1[2].item()
            self.loss_G_triplet_ves_neg = dis2[2].item()

            self.loss_G_triplet = loss_G_triplet * opt.lambda_triplet
            self.loss_G += self.loss_G_triplet

        # centerness scores maps prediction
        if opt.use_centerness:
            
            # use centerness loss
            # 1. mask out the background first
            if self.centerness_block_num == 1:
                self.centerness_maps = self.centerness_maps * self.label_input_sm#self.label_nodisk_map 
            if self.centerness_block_num == 2:
                self.centerness_maps = self.centerness_maps * torch.cat([self.label_input_sm, self.label_dilation_map2], dim=1)
            if self.centerness_block_num == 3:
                self.centerness_maps = self.centerness_maps * torch.cat([self.label_input_sm, self.label_dilation_map2, self.label_dilation_map3], dim=1)

            if opt.center_loss_type == 'centerness': 
                #calculate V, the number of pixel
                #bs, ch, h, w = self.label_input.shape
                #v = bs*ch*h*w
                v1 = torch.sum(self.label_input_sm)
                v2 = torch.sum(self.label_dilation_map2) if self.centerness_block_num >= 2 else 0
                v3 = torch.sum(self.label_dilation_map3) if self.centerness_block_num >= 3 else 0
                self.loss_G_centerness  = 0
                self.loss_G_centerness, self.loss_G_center1, self.loss_G_center2, self.loss_G_center3  = centernessLoss(self.criterionSmoothL1, self.centerness_maps, self.label_centerness_map_all,  v1,v2,v3, weight_list=opt.lambda_dilation_list)

            elif opt.center_loss_type == 'smoothl1':
                # only use smooth l1 loss
                self.loss_G_centerness = self.criterionSmoothL1_mean(self.centerness_maps, self.label_centerness_map_all)
        
            self.loss_G += self.loss_G_centerness 

        # backward
        if isBackward:
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.use_GAN:
            # update D
            set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.scheduler_D.step(self.step)
        # update G
        if self.use_GAN:
            set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.scheduler_G.step(self.step)