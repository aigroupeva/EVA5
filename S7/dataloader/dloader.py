#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:19:20 2020

@author: abhinav

"""

from torch.utils.data import DataLoader

class dloader(DataLoader):
    """
    Base class for all data loaders, support for custom sampler and collate function added
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers,pin_memory):

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory':pin_memory                     
        }

        super().__init__(**self.init_kwargs)