
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np

# Image_Resize
class model(nn.Module):
    def __init__(self, size_out):
        super(Image_Resize, self).__init__()
        self.size_out = size_out
    def forward(self, x):
        x = x.numpy()
        batch, in_dim, in_x, in_y = np.shape(x)
        out_dim, out_x, out_y = self.size_out
        if in_dim == 1 and out_dim == 1:
            tmp = np.zeros((batch, out_dim, out_x, out_y))
            for i in range(0,batch):
                tmp[i, 0, :, :] = cv2.resize(x[i, 0, :, :], (out_x, out_y))  
        if in_dim == 1 and out_dim == 3:
            tmp = np.zeros((batch, out_dim, out_x, out_y))
            for i in range(0, batch):
                a = cv2.resize(np.float32(x[i, 0, :, :]), (out_x, out_y))
                tmp[i, :, :, :] = cv2.cvtColor(cv2.resize(np.float32(
                    x[i, 0, :, :]), (out_x, out_y)), cv2.COLOR_GRAY2RGB).reshape((out_dim, out_x, out_y))
        if in_dim == 3 and out_dim == 1:
            tmp = np.zeros((batch, out_dim, out_x, out_y))
            for i in range(0, batch):
                tmp[i, 0, :, :] = cv2.cvtColor(cv2.resize(np.float32(
                    x[i, :, :, :].reshape((in_x, in_y, in_dim))), (out_x, out_y)), cv2.COLOR_RGB2GRAY).reshape((out_dim, out_x, out_y))
        if in_dim == 3 and out_dim == 3:
            tmp = np.zeros((batch, out_dim, out_x, out_y))
            for i in range(0, batch):
                for j in range(0, out_dim):
                    tmp[i, j, :, :] = cv2.resize(x[i, j, :, :], (out_x, out_y))
        x = torch.from_numpy(tmp)
        return x
