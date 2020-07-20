
import torch

def Loss_Value_Computation(loss_type):
    if loss_type == 'MSELoss':
        loss = torch.nn.MSELoss()
    return loss
