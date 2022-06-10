
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import numpy as np

__all__ = ['Encoding', 'EncodingDrop']

class Encoding(Module):
    """
    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`
    """
    
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        part0_0=torch.matmul(X.unsqueeze(3),self.scale.unsqueeze(0))
        part0_1=part0_0.sum(2)
        part1=(self.scale.view(self.K,1)*self.codewords).sum(1)
        SL=(part0_1-part1)*(part0_1-part1)/self.scale.view(1,self.K)
        A = F.softmax(SL, dim=2)
        # aggregate
        part2=(A.unsqueeze(3)*X.unsqueeze(2)).sum(1)
        part3=(A.unsqueeze(3)*self.codewords.view(1,1,self.K,self.D)).sum(1)
        E=part2-part3

        print('part0_0: ', part0_0.size())  #B*M*C*K
        print('part0_1: ', part0_1.size())  #B*M*K
        print('part1: ', part1.size())      #K
        print('SL: ', SL.size())            #B*M*K
        print('A: ', A.size())              #B*M*K
        print('part2: ', part2.size())      #B*K*C
        print('part3: ', part3.size())      #B*K*C
        print('E: ', E.size())              #B*K*C

        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'
        
class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)



