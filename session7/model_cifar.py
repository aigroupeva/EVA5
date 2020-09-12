from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt

import regularization

class Net(nn.Module):
    def __init__(self,dropout_value = 0):
        super(Net, self).__init__()
    
        
        ## CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), stride=1, padding=1),
#             nn.Conv2d(in_channels=3, out_channels=33, kernel_size=(3, 3), padding=1, groups = 3, bias=False),
#             nn.Conv2d(in_channels=33, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 3
        
        self.convblock2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, groups = 32, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 32 output_size = 32 receptive_field = 5
        

        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 32 output_size = 16 receptive_field = 10
        
        
        ## CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1,dilation = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 14
        
        self.convblock4 = nn.Sequential(
            # Dilated Convolution of 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, groups = 32, bias=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) # input_size = 16 output_size = 16 receptive_field = 16
        
        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 16 output_size = 8   receptive_field = 32
        
        
        ## CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 35       
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation = 1, groups = 32, bias=False),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 6   output_size = 6 receptive_field = 39 
        
        ## CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),
            nn.ReLU()
        ) # input_size = 8   output_size = 8 receptive_field = 43
        
#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0, dilation = 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.Dropout(dropout_value),
#             nn.ReLU()
#         ) # input_size = 8   output_size = 6  receptive_field = 45
        
        
        
        self.gap = nn.AvgPool2d(kernel_size=(6,6))        
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.convblock2(self.convblock1(x)))
        x = self.pool2(self.convblock4(self.convblock3(x)))
        x = self.convblock6(self.convblock5(x))
        x = self.convblock7(x)
#         x = self.convblock8(self.convblock7(x))
        x = self.gap(x)
#         print(x.shape)
        x = x.view(-1, 256)
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=-1)