# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:32:08 2023

@author: khoda
"""


import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import csv
import scipy.signal
#from dataset_utils import get_config
#from mmcv.utils import Config
from dataset_utils import SSL_config
import torch

config_paths = {
    'data_droplet':'Dataset/config.py',
    'train_droplet':'train_config.py'
    }

class unsupervised_segmentation():
    def __init__(self,config):
        super().__init__()
        self.config = config
        
    def crop(self,left_crop=0,k=3,morphology_kernel_size = 35,median_filter_kernel=3,visualization=True):
        brightSpot_indices = {}
        Area_list = {}
        Diameter_list = {}
        segmented_img_list = {}
        real_img_list = {}
        org_img_list = {}
        y2_list = {}
        
        dataset_config = SSL_config.get_config(self.config)
        folder_path = dataset_config.folder_path
        folders = os.listdir(folder_path)
        for folder in folders:
            print("folder:",folder)
            Area_list[folder] = list()
            Diameter_list[folder] = list()
            org_img_list[folder] = list()
            segmented_img_list[folder] = {}
            brightSpot_indices[folder] = {}
            real_img_list[folder] = {}
            
            
            images = sorted(os.listdir(folder_path+'/'+folder))
            gaussian_kernel = (5,5)
            matplotlib.use('Qt5Agg')
            Tube_dia,Tube_length_pixels, y2 = SSL_config.tube_scale(config = dataset_config,folder=folder)
            Tube_dia = float(Tube_dia)
            Tube_length_pixels = float(Tube_length_pixels)
            y2 = float(y2)
            y2_list[folder] = y2
            matplotlib.use('module://ipykernel.pylab.backend_inline')
            
            up_crop = dataset_config.up_crop[folder]
            crop_W = dataset_config.crop_W[folder]
            
            for image in images:
                if image.endswith('.tif'):
                    print(image)
                    img = cv2.imread(folder_path +'/'+folder+'/'+ image)
                    # Apply Gaussian filter if needed
        #             img = cv2.GaussianBlur(img,gaussian_kernel,0)
        
                    # Find the estimated location of the droplet
                    cropped_img = img[int(y2) + up_crop:,left_crop:]
                    cropped_img = cropped_img[:,:,0]
                    ind = np.unravel_index(np.argmax(cropped_img), cropped_img.shape)
                    
                    if ind[1] >crop_W & ind[0]>crop_W:
                        cropped_img = cropped_img[ind[0]-crop_W:ind[0]+crop_W,ind[1]-crop_W:ind[1]+crop_W]
                    elif ind[1]<crop_W & ind[0]>crop_W:
                        cropped_img = cropped_img[ind[0]-crop_W:ind[0]+crop_W,0:2*crop_W]
                    elif ind[1]>crop_W & ind[0]<crop_W:
                        cropped_img = cropped_img[0:2*crop_W,ind[1]-crop_W:ind[1]+crop_W]
        
                    else:
                        cropped_img = cropped_img[0:2*crop_W,0:2*crop_W]
                        
                    # Run the following for inside images  
                    cropped_img = img[int(y2)+up_crop+ind[0]-crop_W:int(y2)+up_crop+ind[0]+crop_W,left_crop+ind[1]-crop_W:left_crop+ind[1]+crop_W]
                    print(cropped_img.shape)
                    try:
                        flattened_img = cropped_img.reshape(-1,1)
                        flattened_img = np.float32(flattened_img)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
                        _, labels,centers = cv2.kmeans(flattened_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
                        centers = np.uint8(centers)
                        labels = labels.flatten()
        
                        labels_occ = np.bincount(labels)
                        background_label = np.argmax(labels_occ)
        
                        segmented_image = centers[labels.flatten()]
                        segmented_image[labels==background_label] = 0
                        segmented_image[labels!=background_label] = 255
                        segmented_image = segmented_image.reshape(cropped_img.shape)
        
                        #Median filter to remove noisy white pixels
                        smoothed_img = cv2.medianBlur(segmented_image, median_filter_kernel)
        
                        kernel = np.ones((morphology_kernel_size,morphology_kernel_size))
                        # do a morphologic close to fill out the droplet
                        filled_img = cv2.morphologyEx(smoothed_img,cv2.MORPH_CLOSE, kernel)
                        if visualization:
                            fig,ax = plt.subplots(1,2)
                            ax[0].imshow(cropped_img,cmap='gray')
                            ax[1].imshow(filled_img,cmap='gray')
        
                        area = np.count_nonzero(filled_img)
                        Real_area = area*(Tube_dia*Tube_dia)/(Tube_length_pixels*Tube_length_pixels)
                        Real_diameter = 2*np.sqrt(Real_area/3.14)
        
                        Area_list[folder].append(Real_area)
                        Diameter_list[folder].append(Real_diameter)
                        segmented_img_list[folder][image] = filled_img
                        plt.show()
                        
                        real_img_list[folder][image] = cropped_img
                        org_img_list[folder].append(image)
                        brightSpot_indices[folder][image]= ind
                    except:
                        image_problem = 0
                        print("There is a problem with the image")
                
        return Area_list,Diameter_list,segmented_img_list,real_img_list,brightSpot_indices,org_img_list,y2_list
        
    def save_crop(self,real_img_list,segmented_img_list):
        
        dataset_config = SSL_config.get_config(self.config)
        folder_path = dataset_config.folder_path
        save_dir = dataset_config.save_dir
        seg_save_dir = dataset_config.seg_save_dir
        folders = os.listdir(folder_path)
        
        for folder in folders:
            real_img_folder = real_img_list[folder]
            segmented_img_folder = segmented_img_list[folder]
        
            for i in list(real_img_folder.keys()):
                cv2.imwrite(save_dir+folder+f"/{i.replace('.tif','')}.png",real_img_folder[i])
                cv2.imwrite(seg_save_dir+folder+f"/{i.replace('.tif','')}.png",segmented_img_folder[i])

    
    def coordinate_mapping(self,org_img_list,segmented_img_list,y2_list,brightSpot_indices):
        full_seg_list = {}
        full_img_list = {}
        dataset_config = SSL_config.get_config(self.config)
        folder_path = dataset_config.folder_path
        folders = os.listdir(folder_path)
        seg_save_dir = dataset_config.seg_save_dir
        
        up_crop = dataset_config.up_crop
        crop_W = dataset_config.crop_W
        left_crop = dataset_config.left_crop
        
        for folder in folders:
            
            for seg in org_img_list[folder]:
                if seg.replace('tif','png') in os.listdir(seg_save_dir+'/'+folder):
                    segmented_img_list[seg] = cv2.imread(seg_save_dir+'/'+folder+seg.replace('tif','png'))
                
        
            n=0
            for image in org_img_list[folder]:
                
                try:
                    full_img = cv2.imread(folder_path +'/'+folder+'/'+ image)
                    full_img = full_img[:,:,0]
                    img = cv2.imread(folder_path +'/'+folder+'/'+ image)
                    img = img[:,:,0]
                    segmented_img = segmented_img_list[folder][image]
                    segmented_img = segmented_img[:,:,0]
                    non_zeros = np.nonzero(segmented_img)
        
                    pixels_org_img_ax0 = [y2_list[folder]+i+up_crop[folder]+brightSpot_indices[folder][image][0]-crop_W[folder] for i in non_zeros[0]]
                    pixels_org_img_ax1 = [i+left_crop[folder]+brightSpot_indices[folder][image][1]-crop_W[folder] for i in non_zeros[1]]
                    org_img_indices = list(zip(pixels_org_img_ax0,pixels_org_img_ax1))
                    n+=1
                    img[:,:] = 0
                    for indice in org_img_indices:
                        img[int(indice[0]),int(indice[1])] = 255
                    full_seg_list[image] = img
                    full_img_list[image] = full_img
                    
                    #Save the full-sized segmentation and images
                    cv2.imwrite(dataset_config.full_seg_dir +f'/{image}', img)
                    cv2.imwrite(dataset_config.full_img_dir+f'/{image}',full_img)
                except:
                    print("Error in finding the image, going to next image.")
        return full_seg_list,full_img_list
        
