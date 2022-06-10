import torch.nn.functional as F
from torch import nn
import torch 
import numpy as np
from torch.autograd import Variable

class SmoothL1_weighted():
    def __init__(self, weight_list=[]):
        self.smoothL1Loss = nn.SmoothL1Loss(reduction='none')
        self.weight_list = weight_list

    def __call__(self, preds, targs):
        bs, ch, h, w = preds.shape
        term = self.smoothL1Loss(preds, targs)
        v = bs * 3 * h * w
        term0 = torch.sum(term[:,0:3,:,:]) / v
        term1 = torch.sum(term[:,3:6,:,:]) / v if ch >= 6 else None
        term2 = torch.sum(term[:,6:9,:,:]) / v if ch >= 9 else None

        loss = term0 * self.weight_list[0]
        if ch >= 6:
            loss += term1 * self.weight_list[1]
        if ch >= 9:
            loss += term2 * self.weight_list[2]
        
        return loss
def centernessLoss(criterion, centerness_maps, label_centerness_map, v1,v2,v3, weight_list=[], epsilon=1e-12):
    # 2. calculate smooth l1
    smoothl1 = criterion(centerness_maps, label_centerness_map)
    # print("smoothl1:",smoothl1.shape, smoothl1.max(),smoothl1.min())
    # 3. calculate the square of predicted centerness map
    square_centerness_map = label_centerness_map + epsilon
    # square_centerness_map = label_centerness_map * label_centerness_map + epsilon
    # print("square_centerness_map:", square_centerness_map.shape, square_centerness_map.max(), square_centerness_map.min())
    # 4. calculate the 1/S^2 * smoothL1(S,S')
    term1 = smoothl1 / square_centerness_map

    bs, ch, h, w = centerness_maps.shape

    term1_0 = term1[:,0:3,:,:]
    term1_1 = term1[:,3:6,:,:] if ch >= 6 else None
    term1_2 = term1[:,6:9,:,:] if ch >= 9 else None

    term1_sum0 = torch.sum(term1_0)
    term1_sum1 = torch.sum(term1_1) if ch >= 6 else None
    term1_sum2 = torch.sum(term1_2) if ch >= 9 else None

    # 6. loss
    loss  = 0
    loss1 = term1_sum0 / v1 * weight_list[0]
    if ch >= 6:
        loss2 = term1_sum1 / v2 * weight_list[1]
    if ch >= 9:
        loss3 = term1_sum2 / v3 * weight_list[2]
    loss = loss1 + loss2 + loss3

    return loss, loss1,loss2,loss3

def tripletMarginLoss_vggfea(vggnet, criterion, preds, targs, targs_shuffle, use_cuda=False, anchor='real', weight_list=[1,1,1]):
    '''
        achor : real or fake
    '''
    loss = 0
    # artery
    feat_pred_a = vggnet(torch.cat([preds[:, 0:1], preds[:, 0:1], preds[:, 0:1]], dim=1))
    feat_label_a = vggnet(torch.cat([targs[:, 0:1], targs[:, 0:1], targs[:, 0:1]], dim=1))
    feat_label_sf_a = vggnet(torch.cat([targs_shuffle[:, 0:1], targs_shuffle[:, 0:1], targs_shuffle[:, 0:1]], dim=1))
    # vein
    feat_pred_v = vggnet(torch.cat([preds[:, 2:], preds[:, 2:], preds[:, 2:]], dim=1))
    feat_label_v = vggnet(torch.cat([targs[:, 1:2], targs[:, 1:2], targs[:, 1:2]], dim=1))
    feat_label_sf_v = vggnet(torch.cat([targs_shuffle[:, 1:2], targs_shuffle[:, 1:2], targs_shuffle[:, 1:2]], dim=1))
    # vessel
    feat_pred_ves = vggnet(torch.cat([preds[:, 1:2], preds[:, 1:2], preds[:, 1:2]], dim=1))
    feat_label_ves = vggnet(torch.cat([targs[:, 2:], targs[:, 2:], targs[:, 2:]], dim=1))
    feat_label_sf_ves = vggnet(torch.cat([targs_shuffle[:, 2:], targs_shuffle[:, 2:], targs_shuffle[:, 2:]], dim=1))

    N = len(feat_pred_a)
    dis1 = torch.tensor([0,0,0]).cuda() if use_cuda else torch.tensor([0,0,0]) # a,v,vessel
    dis2 = torch.tensor([0,0,0]).cuda() if use_cuda else torch.tensor([0,0,0])
    dis1 = dis1.float()
    dis2 = dis2.float()
    for i in range(N):
        bs, num = feat_pred_a[i].shape
        num_elem = bs # bs * num
        if anchor == 'real':
            dis1[0] += torch.norm(feat_label_a[i] - feat_pred_a[i], 2) / num_elem
            dis2[0] += torch.norm(feat_label_a[i] - feat_label_sf_a[i], 2) / num_elem

            dis1[1] += torch.norm(feat_label_v[i] - feat_pred_v[i], 2) / num_elem
            dis2[1] += torch.norm(feat_label_v[i] - feat_label_sf_v[i], 2) / num_elem

            dis1[2] += torch.norm(feat_label_ves[i] - feat_pred_ves[i], 2) / num_elem
            dis2[2] += torch.norm(feat_label_ves[i] - feat_label_sf_ves[i], 2) / num_elem

            loss += criterion(feat_label_a[i], feat_pred_a[i], feat_label_sf_a[i]) + \
               criterion(feat_label_v[i], feat_pred_v[i], feat_label_sf_v[i]) + \
               criterion(feat_label_ves[i], feat_pred_ves[i], feat_label_sf_ves[i])
        elif anchor == 'fake':
            dis1[0] += torch.norm(feat_pred_a[i] - feat_label_a[i], 2) / num_elem
            dis2[0] += torch.norm(feat_pred_a[i] - feat_label_sf_a[i], 2) / num_elem

            dis1[1] += torch.norm(feat_pred_v[i] - feat_label_v[i], 2) / num_elem
            dis2[1] += torch.norm(feat_pred_v[i] - feat_label_sf_v[i], 2) / num_elem

            dis1[2] += torch.norm(feat_pred_ves[i] - feat_label_ves[i], 2) / num_elem
            dis2[2] += torch.norm(feat_pred_ves[i] - feat_label_sf_ves[i], 2) / num_elem

            loss += criterion(feat_pred_a[i], feat_label_a[i], feat_label_sf_a[i]) * weight_list[0] + \
               criterion(feat_pred_v[i], feat_label_v[i], feat_label_sf_v[i]) * weight_list[1] + \
               criterion(feat_pred_ves[i], feat_label_ves[i], feat_label_sf_ves[i]) * weight_list[2]

    dis1 = dis1 / N
    dis2 = dis2 / N
    return loss, dis1, dis2


def vggloss(vggnet, criterion, preds, targs):
    loss = 0
    # artery
    feat_pred_a = vggnet(torch.cat([preds[:, 0:1], preds[:, 0:1], preds[:, 0:1]], dim=1))
    feat_label_a = vggnet(torch.cat([targs[:, 0:1], targs[:, 0:1], targs[:, 0:1]], dim=1))
    # vein
    feat_pred_v = vggnet(torch.cat([preds[:, 2:], preds[:, 2:], preds[:, 2:]], dim=1))
    feat_label_v = vggnet(torch.cat([targs[:, 1:2], targs[:, 1:2], targs[:, 1:2]], dim=1))
    # vessel
    feat_pred_ves = vggnet(torch.cat([preds[:, 1:2], preds[:, 1:2], preds[:, 1:2]], dim=1))
    feat_label_ves = vggnet(torch.cat([targs[:, 2:], targs[:, 2:], targs[:, 2:]], dim=1))

    N = len(feat_pred_a)
    for i in range(N):
        loss += criterion(feat_pred_a[i], feat_label_a[i]) + \
           criterion(feat_pred_v[i], feat_label_v[i]) + \
           criterion(feat_pred_ves[i], feat_label_ves[i])
    return loss

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)

    dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class CrossEntropyLossWithSmooth():
    def __init__(self, smooth, num_classes, use_cuda=False):
        self.nlloss = nn.KLDivLoss()
        self.logSoftmax= nn.LogSoftmax()
        self.smooth = smooth
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.confidence = 1.0 - self.smooth
    def __call__(self, preds, targs):
        assert preds.size(1) == self.num_classes
        smooth_label = torch.ones_like(preds)
        if self.use_cuda:
            smooth_label = smooth_label.cuda()
        smooth_label.fill_(self.smooth / (self.num_classes - 1))
        smooth_label.scatter_(1, targs.data.unsqueeze(1), self.confidence)
        smooth_label = Variable(smooth_label, requires_grad=False)
        loss = self.nlloss(self.logSoftmax(preds), smooth_label)
        return loss

class L1LossWithLogits():
    def __init__(self):
        self.l1loss = nn.L1Loss()

    def __call__(self, preds, targs):
        preds = torch.sigmoid(preds)
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        loss = self.l1loss(preds[:,0], target_artery) + \
               self.l1loss(preds[:,2], target_vein) + \
               self.l1loss(preds[:,1], target_all)
        return loss

class multiclassLoss():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes
        # self.logitsLoss = nn.BCEWithLogitsLoss()
        self.logitsLoss = nn.BCELoss()
        
    def __call__(self, preds, targs):
        #print(preds.shape, targs.shape)
        
#        target_artery = (targs == 2).float()
#        target_vein = (targs == 1).float()
#        target_all = (targs >= 1).float()
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        arteryWeight = 2
        veinWeight = 2
        vesselWeight = 3        

        loss = ( arteryWeight*self.logitsLoss(preds[:,0], target_artery) + 
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
        return loss


