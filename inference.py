# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:17:53 2023

@author: SiaK4
"""
from dataset_utils import SSL_config
import os
from SSL_DataLoader import Drop_Dataset
from torch.utils.data import DataLoader
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import albumentations as A
matplotlib.use('module://ipykernel.pylab.backend_inline')

#n_cpu = os.cpu_count()
n_cpu = 1                
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_predictions(data_type,split):
    
    dataset_config = SSL_config.get_config('data_droplet')
    train_config = SSL_config.get_config('train_droplet')
    #x_test_dir = dataset_config.dataloader_test[data_type]+'images'
    #y_test_dir = dataset_config.dataloader_test[data_type]+'segmentations'
    
    
    val_transform = SSL_config.transform(split='Val')
    test_iou = []
    test_dataset = Drop_Dataset(config='data_droplet',data=data_type,split=split,augmentation = val_transform)
    test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False,num_workers=n_cpu)
    epoch = train_config.num_epochs
    
    model = smp.create_model(
            "Unet",encoder_name=train_config.model['encoder_name'],encoder_weights=train_config.model['encoder_weights'],in_channels=train_config.model['in_channels'], classes=train_config.model['classes'])
    model = model.to(device=DEVICE)
    model.load_state_dict(torch.load('U-Net_checkpoint_epoch'+f"{epoch}"+'.pth'))    
    for x,y in test_dataloader:
        x = x.float().to(device=DEVICE)
        y = y/255
        y = y.float().to(device=DEVICE)
        y_predictions = model(x)
        
        test_prob_mask = y_predictions.sigmoid()
        test_pred_mask = (test_prob_mask>0.5).float()
        #Calculate the iou of the test images
        tp,fp,fn,tn = smp.metrics.get_stats(test_pred_mask.long(),y.long(),mode='binary')
        iou = tp/(tp+fp+fn)
        test_iou.append(iou)
    print("Mean IoU over the test dataset:",torch.mean(sum(test_iou)))

    return test_iou,test_pred_mask,test_dataset

def visualize_predictions(data_type,split='Val',img_h=1920,img_w=1080,gt_mask=True,IMG_DIR=None):
    dataset_config = SSL_config.get_config('data_droplet')
    if gt_mask==True:
        _,test_pred_mask,test_dataset = generate_predictions(data_type,split=split)
        for i in range(len(test_dataset)):
            image, gt_mask = test_dataset[i]
            image_re = np.zeros((image.shape[1],image.shape[2],3))
            image_re[:,:,0] = image[0,:,:]
            image_re[:,:,1] = image[1,:,:]
            image_re[:,:,2] = image[2,:,:]
            gt_mask = gt_mask.reshape(gt_mask.shape[1],gt_mask.shape[2])
            test_image_mask = test_pred_mask[i]
            test_image_mask = test_image_mask.reshape(test_image_mask.shape[1],test_image_mask.shape[2]).detach().cpu()
            
            image_re = cv2.resize(image_re, (img_h,img_w),interpolation = cv2.INTER_AREA)
            gt_mask = cv2.resize(gt_mask, (img_h,img_w),interpolation = cv2.INTER_AREA)
            test_image_mask = cv2.resize(test_image_mask.numpy(), (img_h,img_w),interpolation = cv2.INTER_AREA)
            fig,ax = plt.subplots(1,3,figsize=(10,10))
            ax[0].imshow(image_re)
            ax[0].set_title('Test Image')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].imshow(gt_mask,cmap='gray')
            ax[1].set_title('GT mask')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].imshow(test_image_mask,cmap='gray')
            ax[2].set_title('Predicted mask')
            ax[2].set_xticks([])
            ax[2].set_yticks([])
    elif gt_mask==False:
        train_config = SSL_config.get_config('train_droplet')
        #NEW_IMG_DIR = dataset_config.dataloader_test[data_type]+'images'
        NEW_IMG_DIR = IMG_DIR
        IMG_list = os.listdir(NEW_IMG_DIR)
        
        #Getting the original size for scale transformation
        size_check1 = cv2.imread(NEW_IMG_DIR+IMG_list[0]).shape[0]
        size_check2 = cv2.imread(NEW_IMG_DIR+IMG_list[0]).shape[1]
        scale_transform = A.Compose(
            A.augmentations.geometric.resize.Resize(size_check1,size_check2,interpolation=1,always_apply=True))
        val_transform = SSL_config.transform(split='Val')
        
        model = smp.create_model(
            "Unet",encoder_name=train_config.model['encoder_name'],encoder_weights=train_config.model['encoder_weights'],in_channels=train_config.model['in_channels'], classes=train_config.model['classes'])
        model = model.to(device=DEVICE)
        epoch = train_config.num_epochs
        model.load_state_dict(torch.load('U-Net_checkpoint_epoch'+f"{epoch}"+'.pth')) 
        for i in range(len(IMG_list)):
            image = cv2.imread(NEW_IMG_DIR+IMG_list[i])
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            preprocessed_image = val_transform(image=image)['image']
            image2 = scale_transform(image=image)['image']
            #Getting the original size for scale transformation
            size_check1 = cv2.imread(NEW_IMG_DIR+IMG_list[i]).shape[0]
            size_check2 = cv2.imread(NEW_IMG_DIR+IMG_list[i]).shape[1]
            scale_transform = A.Compose(
                [A.augmentations.geometric.resize.Resize(size_check1,size_check2,interpolation=1,always_apply=True)])
            if image.shape[0]!=3:
                preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1])
                
                preprocessed_image = np.concatenate((preprocessed_image,)*3, axis=0)
            preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1],preprocessed_image.shape[2])
            preprocessed_image = torch.from_numpy(preprocessed_image).float()
            preprocessed_image = preprocessed_image.to(device=DEVICE)

            prediction_mask = model(preprocessed_image)
            prediction_mask_show = prediction_mask.sigmoid()
            prediction_mask_show = (prediction_mask_show > 0.5).float()

            prediction_mask_show = np.array(prediction_mask_show.reshape(prediction_mask_show.shape[2],prediction_mask_show.shape[3]).detach().cpu())
            prediction_mask_show_rescaled = scale_transform(image = prediction_mask_show)['image']
                
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(image,cmap='gray')
            ax[1].imshow(prediction_mask_show_rescaled,cmap='gray')
            
def droplet_size(IMG_DIR):
    train_config = SSL_config.get_config('train_droplet')
    matplotlib.use('Qt5Agg')
    Area_list = []
    Diameter_list = []
    IMG_list = os.listdir(IMG_DIR)
    sample = cv2.imread(IMG_DIR+IMG_list[0])
        
    Tube_dia = input("Please enter the tube diameter in mm: ")
    Tube_dia = float(Tube_dia)
    plt.figure()
    plt.imshow(sample)
    print('Click on top of the tube')
    clicked = plt.ginput(1,timeout=0,show_clicks=True)
    x1,y1 = clicked[0]
    y1 = float(y1)
    print('Click on bottom of the tube')
    clicked = plt.ginput(1,timeout=0,show_clicks=True)
    x2,y2 = clicked[0]
    y2 = float(y2)
    Tube_length_pixels = np.abs(y1-y2)
    print("Tube Length in Pixels:",Tube_length_pixels)
    plt.close()
        
    model = smp.create_model(
        "Unet",encoder_name=train_config.model['encoder_name'],encoder_weights=train_config.model['encoder_weights'],in_channels=train_config.model['in_channels'], classes=train_config.model['classes'])
    model = model.to(device=DEVICE)
    epoch = train_config.num_epochs
    model.load_state_dict(torch.load('U-Net_checkpoint_epoch'+f"{epoch}"+'.pth')) 
    val_transform = SSL_config.transform(split='Val')
    matplotlib.use('module://ipykernel.pylab.backend_inline')
    for i in range(len(IMG_list)):
        image = cv2.imread(IMG_DIR+IMG_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = val_transform(image = image)['image']

        #Getting the original size for scale transformation of the mask
        size_check1 = cv2.imread(IMG_DIR+IMG_list[i]).shape[0]
        size_check2 = cv2.imread(IMG_DIR+IMG_list[i]).shape[1]
        scale_transform = A.Compose(
            [A.augmentations.geometric.resize.Resize(size_check1,size_check2,interpolation=1,always_apply=True)])

        if image.shape[0]!=3:
            preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1])
            preprocessed_image = np.concatenate((preprocessed_image,)*3, axis=0)

        preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1],preprocessed_image.shape[2])

        preprocessed_image = torch.from_numpy(preprocessed_image).float()
        preprocessed_image = preprocessed_image.to(device=DEVICE)

        prediction_mask = model(preprocessed_image)
        prediction_mask_show = prediction_mask.sigmoid()
        prediction_mask_show = (prediction_mask_show > 0.5).float()

        prediction_mask_show = np.array(prediction_mask_show.reshape(prediction_mask_show.shape[2],prediction_mask_show.shape[3]).detach().cpu())
        prediction_mask_show_rescaled = scale_transform(image = prediction_mask_show)['image']
            
        #Calculating the droplet area and estimated droplet diameter
        droplet_area_pixel = np.count_nonzero(prediction_mask_show_rescaled)
        Real_droplet_area = droplet_area_pixel*(Tube_dia**2)/(Tube_length_pixels**2)
        Real_droplet_diameter = 2*np.sqrt(Real_droplet_area/3.14)
        Area_list.append(Real_droplet_area)
        Diameter_list.append(Real_droplet_diameter)
        
    return Area_list,Diameter_list
    
def velocity_droplet(IMG_DIR,height_crop=950,fps=600,resolution=1920):
    matplotlib.use('Qt5Agg')
    center_x = {}
    center_y = {}
    train_config = SSL_config.get_config('train_droplet')
    
    IMG_list = os.listdir(IMG_DIR)
    print("Number of images:",len(IMG_list))
    sample = cv2.imread(IMG_DIR+IMG_list[0])
    Tube_dia = input("Please enter the tube diameter in mm: ")
    Tube_dia = float(Tube_dia)
    plt.figure()
    plt.imshow(sample)
    print('Click on top of the tube')
    clicked = plt.ginput(1,timeout=0,show_clicks=True)
    x1,y1 = clicked[0]
    y1=float(y1)
    print('Click on bottom of the tube')
    clicked = plt.ginput(1,timeout=0,show_clicks=True)
    x2,y2 = clicked[0]
    y2=float(y2)
    Tube_length_pixels = np.abs(y1-y2)
    print("Tube Length in Pixels:",Tube_length_pixels)
    plt.close()
    size_check1 = cv2.imread(IMG_DIR+IMG_list[0]).shape[0]
    size_check2 = cv2.imread(IMG_DIR+IMG_list[0]).shape[1]
    scale_transform = A.Compose(
        [A.augmentations.geometric.resize.Resize(size_check1,size_check2,interpolation=1,always_apply=True)])
    
    model = smp.create_model(
            "Unet",encoder_name=train_config.model['encoder_name'],encoder_weights=train_config.model['encoder_weights'],in_channels=train_config.model['in_channels'], classes=train_config.model['classes'])
    model = model.to(device=DEVICE)
    epoch = train_config.num_epochs
    model.load_state_dict(torch.load('U-Net_checkpoint_epoch'+f"{epoch}"+'.pth')) 
    for i in range(len(IMG_list)):
        image = cv2.imread(IMG_DIR+IMG_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:height_crop,:]
        val_transform = SSL_config.transform(split='Val')
        preprocessed_image = val_transform(image=image)['image']
        if image.shape[0]!=3:
            preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1])
            preprocessed_image = np.concatenate((preprocessed_image,)*3, axis=0)
        
        preprocessed_image = preprocessed_image.reshape(1,preprocessed_image.shape[0],preprocessed_image.shape[1],preprocessed_image.shape[2])
        preprocessed_image = torch.from_numpy(preprocessed_image).float()
        preprocessed_image = preprocessed_image.to(device=DEVICE)
        
        prediction_mask = model(preprocessed_image)
        prediction_mask_show = prediction_mask.sigmoid()
        prediction_mask_show = (prediction_mask_show > 0.5).float()
        prediction_mask_show = np.array(prediction_mask_show.reshape(prediction_mask_show.shape[2],prediction_mask_show.shape[3]).detach().cpu())
        prediction_mask_show_rescaled = scale_transform(image = prediction_mask_show)['image']
        
        prediction_mask_show_rescaled = cv2.resize(prediction_mask_show_rescaled, (1920,1080),interpolation = cv2.INTER_AREA)
        non_zero_indices = np.nonzero(prediction_mask_show_rescaled)
        y_coor = non_zero_indices[0]
        x_coor = non_zero_indices[1]
        y_coor_center = int((np.max(y_coor) + np.min(y_coor))/2)
        #center_y[int(IMG_list[i].replace(".png",""))] = y_coor_center
        center_y[IMG_list[i]] = y_coor_center
        x_coor_center = int((np.max(x_coor) + np.min(x_coor))/2)
        center_x[IMG_list[i]] = x_coor_center
        
    sorted_keys = sorted(center_y.keys())
    center_y_sorted = {i:center_y[i] for i in sorted_keys}
    
    value_list = list(center_y_sorted.values())
    S = value_list[-1] - value_list[1]
    frames = len(value_list)-1
    
    tube_length = (Tube_dia*image.shape[1])/float(Tube_length_pixels)
    
    time = frames/fps
    v_0 = ((S*tube_length/resolution) - 0.5*9.81*(time**2))/time
    print(v_0)
    
    return v_0