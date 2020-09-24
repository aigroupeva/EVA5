import albumentations as A
# from __future__ import print_function
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch

import model_utility.data_utils as dutils
import model_utility.model_utils as mutils
import model_utility.plot_utils as putils 
import model_utility.regularization as regularization
import model_file.model_cifar as model_cifar

import model_file.models as mod

import matplotlib.pyplot as plt
import seaborn as sns



def L1_Loss_calc(model, factor=0.0005):
    l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0
    for param in model.parameters():
        #zero_vector = torch.rand_like(param)*0
        zero_vector=torch.zeros_like(param)
        reg_loss += l1_crit(param,zero_vector)

    return factor * reg_loss
