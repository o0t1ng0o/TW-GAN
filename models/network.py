# -*- coding: utf-8 -*-
"""

@author: wenting
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50
import torch.nn.functional as F
from models.unet_parts import double_conv,inconv,down,up,outconv
from models.layers import *
import numpy as np
import math
from torchvision import models

class TWGAN_Net(nn.Module):
    def __init__(self,input_ch=3, resnet='resnet34', num_classes=2, use_cuda=False, pretrained=True, centerness=False, centerness_block_num=1, centerness_map_size=[128, 128]):
        super(TWGAN_Net,self).__init__()

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
        if self.centerness and centerness_map_size[0] == 128:
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
        
        if self.centerness and centerness_map_size[0] == 256:
            # block 1 
            self.cenBlock1 = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU()
            ]
            self.cenBlock1 = nn.Sequential(*self.cenBlock1)

            # block 2
            self.cenBlock2 = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),        
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
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
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),        
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU()        
            ]
            self.cenBlock3 = nn.Sequential(*self.cenBlock3)

            # centerness block
            self.cenBlockFinal = [
                nn.Conv2d(32*self.centerness_block_num, 64, kernel_size=1, padding=0), # 96 = 32*3
                nn.BatchNorm2d(64),
                nn.PReLU(),        
                nn.Conv2d(64, 3*self.centerness_block_num, kernel_size=1, padding=0),
                nn.BatchNorm2d(3*self.centerness_block_num),   
                nn.Sigmoid()
            ]
            self.cenBlockFinal = nn.Sequential(*self.cenBlockFinal)
        

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
