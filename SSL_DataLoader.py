# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:13:55 2023

@author: SiaK4
"""
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from dataset_utils import SSL_config
import os
import numpy as np


class Drop_Dataset(BaseDataset):
  def __init__(self,config,data,split,augmentation=None): 
      dataset_config = SSL_config.get_config(config)
      self.data = data
      image_dir = dataset_config.dataloader[self.data]+'/'+split+'/images'
      mask_dir = dataset_config.dataloader[self.data]+'/'+split+'/segmentations'
      self.ids = os.listdir(image_dir)
      self.images_fps = [os.path.join(image_dir, image_id) for image_id in self.ids]
      self.masks_fps = [os.path.join(mask_dir, image_id) for image_id in self.ids]
      self.augmentation = augmentation

  def __getitem__(self, i):
    # read data and convert the image to gray scale
    image = cv2.imread(self.images_fps[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Read the mask and do resizing (if required)
    mask = cv2.imread(self.masks_fps[i],0)
    if self.augmentation == None:
        #Resize the image to (300*300)
        image = cv2.resize(image, (300,300),interpolation = cv2.INTER_AREA)
    
        #Padd around the image so that the final size would be (320*320)
        top = bottom = (320 - image.shape[0])//2
        right = left = (320 - image.shape[1])//2
        image = np.pad(image,pad_width=top,mode='edge')
    
        #Convert image from GRAY to RGB to have 3 channels
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image = np.transpose(image)  #Transpose it so that it look like (3*320*320)
    
        #Resize the mask to (300*300)
        mask = cv2.resize(mask, (300,300),interpolation=cv2.INTER_AREA)
    
        #Pad around the mask so that the final size is (320*320)
        mask = np.pad(mask,pad_width = top,mode='edge')
    
        #Reshape the mask to (1*320*320) (If required)
        # mask = mask.reshape(1,mask.shape[0],mask.shape[1])
        mask = (mask/255).astype('float')  #convert the pixel values to [0,1]

    #If you want to apply augmentation, it should be applied to both the image and the mask
    elif self.augmentation:
        sample = self.augmentation(image = image, mask=mask)
        image, mask = sample['image'], sample['mask']

    # Correcting the image and mask shape so that the final image shape is (3,h,w) and the final mask shape is (1,h,w)
    mask = mask.reshape(1,mask.shape[0],mask.shape[1])
    if image.shape[0]!=3:
        image = image.reshape(1,image.shape[0],image.shape[1])
        image = np.concatenate((image,)*3, axis=0)

    return image, mask
  def __len__(self):
      return len(self.ids)

