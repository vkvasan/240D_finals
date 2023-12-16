import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,act_quantization


class QGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nbits):
        super(QGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,nbits)
        self.gc2 = GraphConvolution(nhid, nclass,nbits)
        
        self.dropout = dropout
        self.bit = nbits
        self.act_alq = act_quantization(self.bit)
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adj):

        adj  = self.act_alq(adj,self.act_alpha)
        adj_scaling_factor = self.act_alpha/(2**(self.bit)-1)
        x = self.gc1(x, adj,adj_scaling_factor)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj,adj_scaling_factor)
        return F.softmax(x, dim=1)


