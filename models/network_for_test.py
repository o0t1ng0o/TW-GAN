# -*- coding: utf-8 -*-
"""

@author: wenting
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50,resnet18_2,resnet18_3,resnet18_4,resnet18_5
import torch.nn.functional as F
from models.unet_parts import double_conv,inconv,down,up,outconv
from models.layers import *
from models.normalization import SPADE
import numpy as np
import math
from torchvision import models

'''useless code'''

class output_block2_weight2(nn.Module):
    def __init__(self, n_in, use_cuda=False, use_spade=False):
        super().__init__()
        self.use_cuda = use_cuda
        self.use_spade = use_spade
        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)

        if self.use_spade:
            self.spade = SPADE('spadesyncbatch3x3', 64, 1)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x1 = F.relu(self.bn1(x1))
        output1 = self.bn_out1(self.conv_out1(x1))
        weight = F.sigmoid(output1)

        x2 = self.conv2(x_in)
        x2 = F.relu(self.bn2(x2))
        x2 = torch.cat([x1, x2], dim=1)

        # weight = F.hardtanh(weight,max_val=0.5)

        # weight = F.threshold(weight-0.5,0,0)

        #        weight = F.threshold(weight,0.5,0)
        #        weight = F.threshold(1-weight,0.5,torch.exp(-torch.mul(weight,weight)))
        # weight = torch.exp(-torch.mul(weight,weight))

        weight = weight - 0.5
        # weight = weight + 1
        # zeros = torch.zeros(weight.shape)
        ones = torch.ones(weight.shape)
        if self.use_cuda:
            ones = ones.cuda()
        sigma = 1
        weight = sigma * torch.exp(-torch.mul(weight, weight)) + 1 - sigma * torch.exp(-ones / 4)
        # weight = torch.exp(-torch.mul(weight,weight))

        # weight = torch.where(weight<0.5,zeros.cuda(),weight_.cuda())

        # weight = torch.where(weight<0,zeros.cuda(),weight_.cuda())

        # weight_[weight<0.5] = 0

        if self.use_spade:
            x2 = self.spade(x2, weight)
        else:
            x2 = x2 * weight

        output2 = self.bn_out2(self.conv_out2(x2))

        output2_a = output2[:, 0].unsqueeze(1)
        output2_v = output2[:, 1].unsqueeze(1)

        output = torch.cat([output2_a, output2_v, output1], dim=1)
        return output


class TRGAN_Net(nn.Module):
    def __init__(self,input_ch=3, resnet='resnet34', num_classes=2, use_cuda=False, pretrained=True, centerness=False, centerness_block_num=1):
        super(TRGAN_Net,self).__init__()

        cut = 7
        self.centerness_block_num = centerness_block_num
        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=pretrained,input_ch=input_ch).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
       
        # final convolutional layers
        # predict artery, vein and vessel
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        self.bn_out = nn.BatchNorm2d(3)

        # use centerness block
        self.centerness = centerness
        if self.centerness :
            # block 1 
            self.cenBlock1 = [
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU()
            ]
            self.cenBlock1 = nn.Sequential(*self.cenBlock1)

            # block 2
            self.cenBlock2 = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU()        
            ]
            self.cenBlock2 = nn.Sequential(*self.cenBlock2)

            # block 3
            self.cenBlock3 = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),        
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU()        
            ]
            self.cenBlock3 = nn.Sequential(*self.cenBlock3)

            # centerness block
            self.cenBlockFinal = [
                nn.Conv2d(64*self.centerness_block_num, 128, kernel_size=1, padding=0), # 192 = 64*3
                nn.BatchNorm2d(128),
                nn.PReLU(),        
                nn.Conv2d(128, 3*self.centerness_block_num, kernel_size=1, padding=0),
                nn.BatchNorm2d(3*self.centerness_block_num),   
                nn.Sigmoid()
            ]
            self.cenBlockFinal = nn.Sequential(*self.cenBlockFinal)
        
        # useless code
        # self.outblock = output_block2_weight2(64, use_cuda=use_cuda, use_spade=False)
        #######################
        #TODO: this is the one witout Channel Attention -- Dict Learning
        #self.out4 = output_block2(64)
        #output_block_encode is for encoding , otherwise, use output_block2
        #######################
        ##TODO: this is the one with CA-Dict Learning.
        self.out4 = output_block_encode(64)
        self.out5 = output_block2_2(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):

        x = F.relu(self.rn(x))
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x_f = x

        x_out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        ########################
        # baseline output
        # artery, vein and vessel
        output = self.bn_out(self.conv_out(x_out))

        # use centerness block
        centerness_maps = None
        if self.centerness:
            block1 = self.cenBlock1(self.sfs[0].features)
            # print("feat0:", self.sfs[0].features.shape,"block1:", block1.shape)
            block2 = self.cenBlock2(self.sfs[1].features)
            # print("feat1:", self.sfs[1].features.shape,"block2:", block2.shape)
            block3 = self.cenBlock3(self.sfs[2].features)
            # print("feat2:", self.sfs[2].features.shape,"block3:", block3.shape)

            #blocks = torch.cat([block1, block2, block3], dim=1)
            blocks = [block1]
            if self.centerness_block_num >= 2:
                blocks.append(block2)
                assert len(blocks) == 2
            if self.centerness_block_num >= 3:
                blocks.append(block3)
                assert len(blocks) == 3
            
            blocks = torch.cat(blocks, dim=1)
            # print("blocks", blocks.shape)
            centerness_maps = self.cenBlockFinal(blocks)
            # print("maps:", centerness_maps.shape)
        
        return output, centerness_maps#x_f

    def close(self):
        for sf in self.sfs: sf.remove() 
        
# set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def choose_vgg(name):

    f = None

    if name == 'vgg11':
        f = models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained = True)

    for params in f.parameters():
        params.requires_grad = False

    return f

pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))

class VGGNet(nn.Module):

    def __init__(self, name, layers, cuda = True):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.cuda() if cuda else pretrained_mean
        self.std = pretrained_std.cuda() if cuda else pretrained_std

    def forward(self, x, retn_feats=None, layers=None):

        x = (x - self.mean) / self.std

        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.view(x.shape[0], -1))

        return results


class Ranker(nn.Module):
    def __init__(self, resnet='resnet34', input_ch=3, pretrained=True, num_classes=2, use_cuda=False):
        super(Ranker, self).__init__()

        cut = 8

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(input_ch=input_ch, pretrained=pretrained).children())[:cut]
        base_layers = nn.ModuleList(list(nn.Sequential(*layers)))
        # print("base_layers:", base_layers)
        self.rn = base_layers
        self.avgpool = nn.AvgPool2d(8, stride=1, ceil_mode=True)

        self.fc = nn.Linear(512, num_classes)

        self.mean = pretrained_mean.cuda() if use_cuda else pretrained_mean
        self.std = pretrained_std.cuda() if use_cuda else pretrained_std
        self.use_imgnet_pretrained = pretrained
        if self.use_imgnet_pretrained:
            print("Use model pretrained on ImageNet.")
    def forward(self, x, retn_feats=False, layers=[]):
        # use model trained on imagenet as pretrained model
        if self.use_imgnet_pretrained:
            x = (x - self.mean) / self.std
        if retn_feats:
            feats = []
            for i, model in enumerate(self.rn):
                x = model(x)
                if i in layers:
                    feats.append(x.view(x.shape[0], -1))
            return feats
        x = self.rn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)

        output = self.fc(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

    # set requies_grad=Fasle to avoid computation
