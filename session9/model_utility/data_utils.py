# from __future__ import print_function
import albumentations as A

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


brightness_val =0.13
cantrast_val = 0.1
saturation_val = 0.10
Random_rotation_val = (-7.0, 7.0) 
fill_val = (1,)

normalize_val = (0.5, 0.5, 0.5) #need to give as tuple

def get_data_transform():
    # Train Phase transformation
    train_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness= brightness_val, contrast=cantrast_val, saturation=saturation_val ),
                                          #transforms.RandomRotation = Random_rotation_val, fill=fill_val),
                                          transforms.ToTensor(),
#                                           transforms.Normalize(normalize_val, normalize_val),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
                                            # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                          # Note the difference between (0.1307) and (0.1307,)
                                          ])

    # Test Phase transformation
    test_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          transforms.ToTensor(),
#                                           transforms.Normalize(normalize_val,normalize_val),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                          ])
    # return train transforms test transforms
    return train_transforms, test_transforms


# Check if cuda is available

def get_device():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    print('Device is',device)
    return device



def get_dataset(train_transforms, test_transforms,path):
    trainset = datasets.CIFAR10(path, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(path, train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataloader(batch_size, num_workers, cuda,path):
    
    print("Running over Cuda !! ", cuda)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    train_transforms, test_transforms = get_data_transform()
    trainset, testset = get_dataset(train_transforms, test_transforms,path)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader
