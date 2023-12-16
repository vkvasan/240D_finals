import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import numpy as np

def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)  
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                         
            input_c = input.clamp(min=-1, max=1)     
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i)).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        mean = weight.data.mean()
        std = weight.data.std()
        weight = weight.add(-mean).div(std)      
        weight_q = self.weight_q(weight, self.wgt_alpha)
        
        return weight_q


def act_quantization(b):

    def uniform_quant(x, b=4):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)  
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i)).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply

class GraphConvolution(Module):
   
    def __init__(self, in_features, out_features, nbits, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.bit = nbits
        self.weight_quant = weight_quantize_fn(w_bit=self.bit)
        self.act_alq = act_quantization(self.bit)
        self.act_alq2 = act_quantization(self.bit)
        self.act_alq3 = act_quantization(self.bit)

        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_alpha2 = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_q = Parameter(torch.FloatTensor(in_features, out_features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,adj_scaling_factor):
        weight_q = self.weight_quant(self.weight)       
        self.weight_q = torch.nn.Parameter(weight_q)  
        
        x = self.act_alq(input, self.act_alpha)
        wgt_delta = self.weight_quant.wgt_alpha/(2**(self.bit -1 )-1)
        act_delta = self.act_alpha/(2**(self.bit)-1)

        support = torch.mm(x/act_delta, weight_q/wgt_delta)

        support = support * act_delta * wgt_delta
        support = F.relu(support)

        support_q = self.act_alq2(support, self.act_alpha2)
        support_delta = self.act_alpha2/(2**(self.bit)-1)

        output = torch.mm(adj/adj_scaling_factor, support_q/support_delta)

        return output * adj_scaling_factor* support_delta 
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

