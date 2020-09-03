#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:42:02 2020

@author: abhinav

Functions in this class download and split data 
"""
#libs
from torchvision import datasets

"""`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """


def get_train_test_dataset(train_trans,test_trans):
    train_ds = datasets.MNIST('./', train=True, download=True,transform=train_trans)
    test_ds  = datasets.MNIST('./', train=False, download=True,transform=test_trans)
    return train_ds,test_ds

"""
Based on data set this file will be altered.
Need Solutions for this
"""
