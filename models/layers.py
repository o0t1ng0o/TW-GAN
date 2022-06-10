# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:43:10 2019

@author: wenaoma
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50
import torch.nn.functional as F
from models.unet_parts import double_conv,inconv,down,up,outconv
from models.encoding import Encoding,Mean
import numpy as np

class ConvBn2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(ConvBn2d,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x
    
class scSEBlock(nn.Module):
    def __init__(self,out_channels):
        super(scSEBlock,self).__init__()
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)
    
    def forward(self,x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x+g2*x
        return x

 
class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock0(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        # self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=1, padding=0)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))

#class UnetBlock_down(nn.Module):
#    def __init__(self, up_in, x_in):
#        super().__init__()
#        # super(UnetBlock, self).__init__()
#        self.maxpool = 
#        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
#
#        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
#        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)
#
#        self.bn = nn.BatchNorm2d(n_out)

#    def forward(self, up_p, x_p):
#
#        up_p = self.upsample(up_p)
#        up_p = self.conv1x1(up_p)
#
#        x_p = self.x_conv(x_p)
#        cat_p = torch.cat([up_p, x_p], dim=1)
#        return self.bn(F.relu(cat_p))


class UnetBlock_tran(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))

class UnetBlock_tran_scSE(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.bn(F.relu(cat_p))
        return self.scSE(cat_p)

class UnetBlock_tran3(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return F.relu(self.bn(cat_p))

class UnetBlock_tran2(nn.Module):
    def __init__(self, up_in, x_in):
        super().__init__()
        # super(UnetBlock, self).__init__()
        #up_out = x_out = n_out // 2
        self.x_conv1 = nn.Conv2d(x_in+up_in//2, x_in, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(x_in, x_in, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in//2,kernel_size=2,stride=2)
        #self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(x_in)


    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.x_conv1(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        cat_p = self.x_conv2(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        #up_p = self.conv1x1(up_p)

        return cat_p    
    
class UnetBlock_tran4(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(UnetBlock_tran4, self).__init__()
        self.x_conv1 = nn.Conv2d(x_in, n_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out*2, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,n_out,kernel_size=2,stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        x_p = self.x_conv1(x_p)
        up_p = self.upsample(up_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = F.relu(self.bn(self.x_conv2(cat_p)))
        cat_p = F.relu(self.bn(self.x_conv3(cat_p)))
        
        
        return cat_p 

class UnetBlock_tran5(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.x_conv2 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):


        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv2(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))    
    
class UnetBlock_tran6(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.x_conv2(cat_p)
        return F.relu(self.bn(cat_p))    
    
class UnetBlock_tran6_scSE(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.x_conv2(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        return self.scSE(cat_p)    
    
 

class UnetBlock_tran7_scSE(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.x_conv2(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        return self.scSE(cat_p)    
    
class UnetBlock_tran8_scSE(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)  


class UnetBlock_tran8_scSE_add(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=1, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_in, n_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
#        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)  


class UnetBlock_tran8_scSE_1x1(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, up_in//2, kernel_size=1, padding=0)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_in//2, up_in//2, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_in//2)
        self.bn2 = nn.BatchNorm2d(up_in//2)
        self.scSE = scSEBlock(up_in//2)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)      

class UnetBlock_tran8_scSE_noconv(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_in, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        #x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)       
    
class UnetBlock_tran8(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return cat_p 
    
class UnetBlock_tran8_1x1(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, up_in//2, kernel_size=1, padding=0)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_in//2, up_in//2, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_in//2)
        self.bn2 = nn.BatchNorm2d(up_in//2)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        

        
        return cat_p     
    
class UnetBlock_tran10(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out*2, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.scSE = scSEBlock(up_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return cat_p  
    
class UnetBlock_tran10_bn(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out*2, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.bn3 = nn.BatchNorm2d(up_out)
        self.scSE = scSEBlock(up_out)       

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = F.relu(self.bn3(self.x_conv3(up_p)))
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return cat_p    
    
class UnetBlock_tran10_bn_dp(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out*2, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.bn3 = nn.BatchNorm2d(up_out)
        self.dropout = nn.Dropout(0.5)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = F.relu(self.bn3(self.x_conv3(up_p)))
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = self.dropout(F.relu(self.bn(cat_p)))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return cat_p     
    
    
class UnetBlock_tran10_scSE(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out*2, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.scSE = scSEBlock(up_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return self.scSE(cat_p)  

    
class UnetBlock_tran8_3x3(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in, up_in//2, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_in//2, up_in//2, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_in//2)
        self.bn2 = nn.BatchNorm2d(up_in//2)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return cat_p    
    
    
class UnetBlock_tran8_scSE_bn(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_out, n_out, kernel_size=3, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.bn2 = nn.BatchNorm2d(up_in + x_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = self.x_conv3(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = F.relu(self.bn2(cat_p))
        
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
#        cat_p = self.x_conv2(cat_p)
#        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)      

class UnetBlock_tran9_scSE(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        #self.x_conv3 = nn.Conv2d(up_in, up_in, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_in+x_out, n_out, kernel_size=3, padding=1)

        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)
        self.scSE = scSEBlock(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv2(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        return self.scSE(cat_p)  


    
class expand_compress_block(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)
        self.scSE = scSEBlock(32)

    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.scSE(x)
        x = self.conv_compress(x)
        return F.relu(self.bn_out(x))      

class expand_compress_block2(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)
        self.scSE = scSEBlock(32)

    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.scSE(x)
        x = self.conv_compress(x)
        return self.bn_out(x)   
    
class expand_compress_block3(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.conv_compress(x)
        return F.relu(self.bn_out(x))       
    
    
class expand_compress_block4(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.conv_compress(x)
        return self.bn_out(x)      
    
class expand_compress_block5(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.conv_compress(x)
        return F.sigmoid(self.bn_out(x))       
    
class expand_compress_block6(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=1, padding=0)
        self.conv_ = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn1(x))
        x = self.conv_(x)
        x = F.relu(self.bn2(x))
        x = self.conv_compress(x)
        return self.bn_out(x)   
    

    
    
class expand_compress_block7(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=1, padding=0)
        self.conv_ = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn1(x))
        x = self.conv_(x)
        x = F.relu(self.bn2(x))
        x = self.conv_compress(x)
        return F.relu(self.bn_out(x))    
    
    
class expand_compress_block8(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=1, padding=0)

        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn1(x))
        x = self.conv_compress(x)
        return self.bn_out(x)        
    
    
class output_block1(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)
        

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.relu(self.bn1(x))
        output1 = self.bn_out1(self.conv_out1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        output2 = self.bn_out2(self.conv_out2(x))
        
        output2_a = output2[:,0].unsqueeze(1)
        output2_v = output2[:,1].unsqueeze(1)
        
        output = torch.cat([output2_a,output1,output2_v], dim=1)
        return output   
    
class output_block_encode(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.encoding = nn.Sequential(nn.Conv2d(32, 32, 1,padding=0),nn.BatchNorm2d(32),nn.ReLU(inplace=True),Encoding(D=32, K=16),nn.BatchNorm1d(16),nn.ReLU(inplace=True), Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(32, 32),nn.Sigmoid())
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)

    def forward(self, x_in):
        x1 = self.conv1(x_in) 
        x1 = F.relu(self.bn1(x1))       
        en = self.encoding(x1)
        b, c, _, _ = x1.size()
        en=en.view(en.size(0), -1)
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        x1 = F.relu(x1 + x1 * y)
        #x1 = F.relu(self.bn1(x1))
        output1 = self.bn_out1(self.conv_out1(x1))

        print('encoding2 en: ', en.size())            #B*C
        print('encoding2 gamma: ', gamma.size())      #B*C
        print('encoding2 y: ', y.size())              #B*K*1*1
        print('encoding2 x1: ', x1.size())            #B*K*H*W
        print('encoding2 output1: ', output1.size())  #B*K*H*W

        
        x2 = self.conv2 (x_in)
        x2 = F.relu(self.bn2(x2))
        en2 = self.encoding(x2)
        b2, c2, _, _ = x2.size()
        en2 = en2.view(en.size(0), -1)
        gamma = self.fc(en2)
        y2 = gamma.view(b2, c2, 1, 1)
        x2 = F.relu( x2 * y2)

        x2 = torch.cat([x1,x2], dim=1)

        output2 = self.bn_out2(self.conv_out2(x2))

        
        output2_a = output2[:,0].unsqueeze(1)
        output2_v = output2[:,1].unsqueeze(1)
        
        output = torch.cat([output2_a,output1,output2_v], dim=1)
        return output 
class output_block2(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)
        

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x1 = F.relu(self.bn1(x1))
        output1 = self.bn_out1(self.conv_out1(x1))
        
        x2 = self.conv2(x_in)
        x2 = F.relu(self.bn2(x2))
        x2 = torch.cat([x1,x2], dim=1)
        output2 = self.bn_out2(self.conv_out2(x2))

        
        output2_a = output2[:,0].unsqueeze(1)
        output2_v = output2[:,1].unsqueeze(1)
        
        output = torch.cat([output2_a,output1,output2_v], dim=1)
        return output


class output_block2_2(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
    def forward(self, x_in):
        
        x1 = self.conv1(x_in)
        x1 = F.relu(self.bn1(x1))        

        x2 = self.conv2(x_in)
        x2 = F.relu(self.bn2(x2))
        x2 = torch.cat([x1,x2], dim=1)
        output2 = self.bn_out2(self.conv_out2(x2))
        output2 = self.maxpool(output2)
        
        output2_a = output2[:,0].unsqueeze(1)
        output2_v = output2[:,1].unsqueeze(1)
        
        output = torch.cat([output2_a,output2_v], dim=1)
     
        return output 
    
class output_block3(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(3)

        

    def forward(self, x_in):
        
        x = self.conv1(x_in)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x)) 
        x = self.bn3(self.conv_out(x))
        
        return x       

class output_block4(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 64, kernel_size=3, padding=1)

        self.conv_out = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(3)


        

    def forward(self, x_in):
        
        x = self.conv1(x_in)
        x = F.relu(self.bn1(x))

        x = self.bn2(self.conv_out(x))
        
        return x      
    
class output_block5(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn_out1 = nn.BatchNorm2d(1)
        self.bn_out2 = nn.BatchNorm2d(2)
        

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x1 = F.relu(self.bn1(x1))
        output1 = self.bn_out1(self.conv_out1(x1))
        
        x2 = self.conv2(x_in)
        x2 = F.relu(self.bn2(x2))
        output2 = self.bn_out2(self.conv_out2(x2))

        
        output2_a = output2[:,0].unsqueeze(1)
        output2_v = output2[:,1].unsqueeze(1)
        
        output = torch.cat([output2_a,output1,output2_v], dim=1)
        return output     
