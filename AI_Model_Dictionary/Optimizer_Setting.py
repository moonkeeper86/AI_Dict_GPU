
import torch


def Optimizer_Deploy_Operation(optimizer_type, model, lr, momentum, beta1, beta2, eps):
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer
