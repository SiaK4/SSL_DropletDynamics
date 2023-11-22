# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:31:23 2023

@author: khoda
"""
import torch

reduced_data = True
batch_size = 4
num_epochs= 3
loss = "DiceLoss"

optimizer = {
    'type': torch.optim.Adam,
    'lr': 0.0002
}
scheduler = {
    'type': torch.optim.lr_scheduler.StepLR,
    'step_size': 10,
    'gamma': 0.1
}
model = {
    'encoder_name': "resnet34",
    'encoder_weights': "imagenet",
    'in_channels': 3,
    'classes': 1
}
