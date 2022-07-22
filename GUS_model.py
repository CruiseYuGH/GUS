import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import random

class GUS(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GUS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.normalize_weight.size(1))
        self.normalize_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj


    def gen_A(self, cs, phase, threshold):
        self.cs = cs
        sample = 0
        sample = np.random.rand(1) 
        if phase != "train":
            sample = threshold
        cs[sample < cs] = 1
        cs[sample >= cs] = 0
        _adj = torch.from_numpy(cs).float().cuda()
        _adj = _adj * 0.5/ (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + torch.from_numpy(np.identity(_adj.shape[0], np.int)).float().cuda()
        return _adj


    def forward(self, input, cs, phase, threshold):
        support = torch.matmul(input, self.normalize_weight)
        adj_A = self.gen_adj(self.gen_A(cs, phase, threshold)).cuda()
        self.temp_adjwithgrad = torch.Tensor(adj_A.shape[0], adj_A.shape[0]).requires_grad_()
        self.temp_adjwithgrad.data = adj_A
        output = torch.matmul(self.temp_adjwithgrad, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

               
class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 6, drop_rate = 0.2):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc_6 = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.gus1 = GUS(512, 512)
        self.gus2 = GUS(512, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)



    def forward(self, x, phase, threshold):
        x = self.features(x)
        if self.drop_rate > 0 and phase=="train":
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        cs1 = cosine_similarity(x.cpu().detach())
        x = self.gus1(x, cs1, phase, threshold)
        x = self.relu(x)
        cs2 = cosine_similarity(x.cpu().detach())
        x = self.gus2(x, cs2, phase, threshold)
        out = self.fc_6(x)
        return out, x
      