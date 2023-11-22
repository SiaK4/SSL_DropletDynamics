# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:45:22 2023

@author: khoda
"""
from mmcv.utils import Config
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import albumentations as A
matplotlib.use('Qt5Agg')

config_paths = {
    'data_droplet':'Dataset/dataset_config.py',
    'train_droplet':'train_config.py'
    }
class SSL_config():
    def __init__(self, dataset='data_droplet'):
        super().__init__()
        self.dataset = dataset


    def get_config(dataset):
        config_path = config_paths[dataset]
        config = Config.fromfile(config_path)
        return config
    
    def tube_scale(config,folder):
        Tube_dia = input("Please enter the tube diameter in mm: ")
        print("Tube diameter is: ",Tube_dia)
        folder_path = config.folder_path
        sample = config.sample[folder]
        print("sample:",sample)
        image = cv2.imread(folder_path+'/'+folder+'/'+sample)
        plt.figure()
        plt.imshow(image)
        print('Click on the top of the tube')
        clicked = plt.ginput(1, timeout=0, show_clicks=True)
        x1, y1 = clicked[0]
        print('Click on the bottom of the tube')
        clicked = plt.ginput(1, timeout=0, show_clicks=True)
        x2, y2 = clicked[0]
        print("y2:",y2)
        Tube_length_pixels =np.abs(y1-y2)
        print("Tube length in pixels:",Tube_length_pixels)
        plt.close()
        
        return Tube_dia,Tube_length_pixels, y2
    
    def transform(split='Train'):
        if split=='Train':
            transform = A.Compose(
            [A.CLAHE(),
             A.augmentations.geometric.resize.Resize (320, 320, interpolation=1, always_apply=True),
             # A.Transpose(),
             A.Blur(blur_limit=3),
             A.OpticalDistortion(),
             A.OneOf([A.RandomRotate90(p=0.5),
             A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5)]),
             A.GridDistortion(p=0.2),
             A.ShiftScaleRotate(p=0.25),
             A.GaussNoise(p=0.5),
             A.RandomBrightnessContrast(p=0.5),
             # A.ISONoise(p=0.5),
             A.MultiplicativeNoise(p=0.5),
             A.RandomToneCurve(p=0.5),
             A.Normalize(mean=(0,),std=(1,),p=1)])
            
        else:
            transform = A.Compose(
                        [A.augmentations.geometric.resize.Resize(320,320,interpolation=1,always_apply=True),
                         #  A.Transpose(),
                         A.Normalize(mean=(0,),std=(1,),p=1)])
        return transform
    
    def visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()