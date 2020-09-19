from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def L1_Loss_calc(model, factor=0.0005):
    l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0
    for param in model.parameters():
        #zero_vector = torch.rand_like(param)*0
        zero_vector=torch.zeros_like(param)
        reg_loss += l1_crit(param,zero_vector)

    return factor * reg_loss
