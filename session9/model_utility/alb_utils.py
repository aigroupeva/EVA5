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
from albumentations.pytorch import ToTensor

# al.Cutout()
# al.RandomCrop()
# al.HorizontalFlip()
# al.ElasticTransform()
# al.CoarseDropout()
# al.Normalize()


# albumentations_transform = A.Compose([
#     A.Resize(256, 256), 
#     A.RandomCrop(224, 224),
#     A.HorizontalFlip(),
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     ToTensorV2()
# ])



brightness_val =0.13
cantrast_val = 0.1
saturation_val = 0.10
Random_rotation_val = (-7.0, 7.0) 
fill_val = (1,)

path = os.getcwd()

def find_stats(path):
    mean = []
    stdev = []
    data_transforms = A.Compose([transforms.ToTensor()])
    trainset,testset = get_dataset(data_transforms,data_transforms,path)
    data = np.concatenate([trainset.data,testset.data],axis = 0,out = None)
    data = data.astype(np.float32)/255
    for i in range(data.shape[3]):
        tmp = data[:,:,:,i].ravel()
        print('mean',tmp.mean())
        print('standard dev',tmp.std())
        mean.append(tmp.mean())
#         mean = [i*255 for i in mean]
        stdev.append(tmp.std())
    return mean,stdev

class AlbumCompose():
    def __init__(self, transform=None):
        self.transform = transform
        
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img


def get_data_transform(path):
    mean,stdev = find_stats(path)
    input_size = 32
    train_albumentation_transform = A.Compose([
                                    A.Cutout(num_holes=2,max_h_size=8,max_w_size=8,fill_value=[i*255 for i in mean],always_apply=True,p=0.5),
#                                     A.RandomCrop(height=8,width=8,p=0.020,always_apply=False),
                                    A.HorizontalFlip(p = 0.7,always_apply=True),
#                                     A.ElasticTransform(alpha=1,sigma=50,alpha_affine=10,interpolation=1,border_mode=4,value=None,mask_value=None,always_apply=False,approximate=False,p=0.1),
                                          A.CoarseDropout(max_holes=1,max_height=16,max_width=16,min_holes=None,min_height=4,min_width=4,fill_value=[i*255 for i in mean],always_apply=True,p=0.7,),
                                    A.Normalize(mean=tuple(mean),std=tuple(stdev), max_pixel_value=255,always_apply=True, p=1.0),
                                    A.Resize(input_size,input_size),
                                        ToTensor()])

    # Test Phase transformation
    test_transforms = transforms.Compose([
                                          transforms.ToTensor(),
        transforms.Normalize(tuple(mean),tuple(stdev))
                                          ])
    train_transforms = AlbumCompose(train_albumentation_transform)
#     test_transforms = AlbumCompose(test_transforms)
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


# get_dataloader can now return trainset,testset, train_loader, test_loader

def get_dataloader(batch_size, num_workers, cuda,path ):
    
    print("Running over Cuda !! ", cuda)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_transforms, test_transforms = get_data_transform(path)
    trainset, testset = get_dataset(train_transforms, test_transforms,path)
    

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    
    return trainset, testset, train_loader, test_loader



