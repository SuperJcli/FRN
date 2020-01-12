import torch
import torch.nn as nn
import math

'''
https://arxiv.org/abs/1911.09737
Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks.
'''
class FRN(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(FRN, self).__init__()
        tau = torch.randn(1, requires_grad=True)
        beta = torch.randn((1, channels, 1, 1), requires_grad=True)
        gamma = torch.randn((1, channels, 1, 1), requires_grad=True)

        self.tau = nn.Parameter(tau)
        self.beta = nn.Parameter(beta)
        self.gamma = nn.Parameter(gamma)

        self.register_buffer('mytao', self.tau)
        self.register_buffer('mybeta', self.beta)
        self.register_buffer('mygamma', self.gamma)
        self.eps = eps

    def forward(self, x):
        nu2 = torch.mean(torch.pow(x, 2), dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = torch.max((self.gamma * x + self.beta), self.tau)

        return y