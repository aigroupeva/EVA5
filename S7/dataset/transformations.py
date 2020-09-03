#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:52:36 2020

@author: abhinav
"""

from torchvision import transforms


'''
TO-DO
The transformations should be received by user instead of pre configured settings.
A text or json file is to be made which take in all the transforamti on line by line.
Content of that file will be converted to list and added to train or test data

Transformation File > List of Transformation(Train/Test) > Argument in respective functions 

example, list name is test_arg
return transforms.Compose(test_arg)
'''

# train_transforms = transforms.Compose([
#                                       #  transforms.Resize((28, 28)),
#                                       #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#                                         transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
#                                         ])


# test_transforms = transforms.Compose([
#                                       #  transforms.Resize((28, 28)),
#                                       #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
#                                         ])



# Ultimate function.
# def train_test_transforms():
#     return train_transforms,test_transforms
    
    
# Ultimate function.
# def train_test_transforms(train_transform_list,test_transform_list):
    #train_transform  = transforms.Compose(train_transform_list)
    #test_transform = transforms.Compose(test_transform_list)
    # return train_transform,test_transform