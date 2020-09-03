#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 01:47:40 2020

@author: abhinav

Need to make changes
"""
import torch
from torchsummary import summary


def get_summary(model, input):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = model.to(device)
	return summary(model, input_size=input) 