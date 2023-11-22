# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:57:07 2023

@author: SiaK4
"""
import torch

folder_path = "Dataset/Sample_dataset/"
sample = {'1':'1 (1).tif','2':'2 (1).tif'}
up_crop ={'1':270,'2':270}
crop_W = {'1':150,'2':200}
left_crop = {'1':0,'2':0}
num_classes = 2 # Number of segmentation classes 
in_channels = 1  # Number of input channels (e.g. 3 for RGB data)
class_labels = ["BG", "Droplet"]
img_suffix='.tiff'
save_dir = 'Dataset/Cropped/'
seg_save_dir = 'Dataset/segmentation/'

full_img_dir = 'Dataset/full_images/'
full_seg_dir = 'Dataset/full_segmentations/'

dataloader = {'reduced_data':'Dataset/Reduced_data/',
              'full_data':'Dataset/full_data/'}

dataloader_test = {'reduced_data':'Dataset/Test/reduced_data/',
                   'full_data':'Dataset/Test/full_data/'}


