
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# RNN_Base_Block  最原始的RNN块
class RNN_Base_Block(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, device='cuda'): 
        super(RNN_Base_Block, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.i2h = nn.Linear(in_channels + hidden_size, hidden_size)
        self.i2o = nn.Linear(in_channels + hidden_size, out_channels)
        self.tanh = nn.Tanh()       # Tanh激活函数
        self.softmax = nn.Softmax()
    def forward(self, x, hidden=None):
        sp = np.shape(x)
        if len(sp)==1:
            x = x.view([sp[0], -1])
        elif len(sp) == 4:
            x = x.view([sp[0], sp[2]*sp[3]])
        batch_size = x.shape[0]
        if hidden is None:
            # hidden ->[batch_size,hidden_size]
            hidden = Variable(torch.zeros(
                batch_size, self.hidden_size)).to(self.device)
        # x:[N,26]+[N,hidden_size]=N,26+hidden_size -> N*hidden_size
        combined = torch.cat((x, hidden), 1)
        # N*hidden_size，这里计算了一个hidden，hidden会带来下一个combined里
        combined = self.tanh(combined)
        hidden = self.i2h(combined)
        out = self.i2o(combined) 
        out = self.softmax(out)
        return out,hidden

# RNN_Ori_iSequential  非序列对象的原始RNN
class RNN_Ori_iSequential(nn.Module):
    input_resize = (28,28)
    def __init__(self, in_channels, hidden_size, R_times=10, num_classes=10, device='cuda'):
        super(RNN_Ori_iSequential, self).__init__()
        self.R_times = R_times
        self.RNN_layer = RNN_Base_Block(in_channels, hidden_size, num_classes, device)
    def forward(self, x, hidden=None):
        if self.R_times == 1:
            out = self.RNN_layer(x, hidden)
        elif self.R_times > 2:
            for i in range(self.R_times-1):
                out, hidden = self.RNN_layer(x, hidden)
            out, hidden = self.RNN_layer(x, hidden)
        return out

# RNN_Ori_Sequential  序列对象的原始RNN
class RNN_Ori_Sequential(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, hidden_size, num_classes=100, device='cuda'):
        super(RNN_Ori_Sequential, self).__init__()
        self.RNN_layer = RNN_Base_Block(
            1, hidden_size, num_classes, device)
        self.hg = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Linear(num_classes, 1),
            #nn.Sigmoid(),  # 20200718此处加这个为了数值回归，不加则为分类问题
        )
    def forward(self, x, hidden=None):
        if x.size()[1] == 1:
            out = self.RNN_layer(x, hidden)
        elif x.size()[1] > 2:
            for i in range(x.size()[1]):
                x_ = x[:,i]
                out, hidden = self.RNN_layer(x_, hidden)
            out = self.hg(out)             #20200718此处加这个为了数值回归，不加则为分类问题
        return out
