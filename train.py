# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:54:03 2023

@author: SiaK4
"""
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from dataset_utils import SSL_config
import segmentation_models_pytorch as smp
import torch
from SSL_DataLoader import Drop_Dataset
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#n_cpu = os.cpu_count()                
n_cpu = 1

class DropNet():
    def __init__(self,config,data):
        super().__init__()
        self.config = config
        self.data = data
        
        train_config = SSL_config.get_config('train_droplet')
        
        encoder_name = train_config.model['encoder_name']
        encoder_weights = train_config.model['encoder_weights']
        in_channels = train_config.model['in_channels']
        classes = train_config.model['classes']
        
        self.model = smp.create_model(
            "Unet", encoder_name=encoder_name, encoder_weights = encoder_weights,in_channels=in_channels, classes=classes)
        
        self.loss_fn = smp.losses.__dict__[train_config['loss']](smp.losses.BINARY_MODE,from_logits=True)
        self.optimizer = train_config.optimizer['type'](self.model.parameters(),lr=train_config.optimizer['lr'])
        self.scheduler = train_config.scheduler['type'](self.optimizer, train_config.scheduler['step_size'],train_config.scheduler['gamma'])
        
        dataset_config = SSL_config.get_config('data_droplet')        
        train_transform = SSL_config.transform(split='Train')
        val_transform = SSL_config.transform(split='Val')
        self.train_dataset = Drop_Dataset(config='data_droplet',data=self.data,split='Train',augmentation = train_transform)
        self.val_dataset = Drop_Dataset(config='data_droplet',data=self.data,split='Val',augmentation = val_transform)
        
        self.NUM_EPOCHS = train_config.num_epochs
        
        self.train_loader = DataLoader(self.train_dataset,train_config.batch_size,shuffle=True,num_workers=n_cpu)
        self.val_loader = DataLoader(self.val_dataset,len(self.val_dataset),shuffle=False,num_workers=n_cpu)
        
    def train_oneEpoch(self):
        IoU = []
        loss_list = []
        
        for batch_idx, (image_data,target_mask) in enumerate(self.train_loader):
            image_data = image_data.float().to(device=DEVICE)
            target_mask = target_mask/255
            target_mask = target_mask.float().to(device=DEVICE)
            
            #forward pass
            #Normalize the image if needed
            #data = (data - torch.mean(data,axis=(2,3),keepdims=True))/(torch.std(data,axis=(2,3),keepdims=True))
            
            predictions = self.model(image_data)
            #Convert mask values to probabilities
            prob_mask = predictions.sigmoid()
            #Then, apply thresholding
            pred_mask = (prob_mask>0.5).float()
            
            #Calculate the loss
            loss = self.loss_fn(predictions,target_mask)
            
            #Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                #Check the accuracy over the train dataset
                tp,fp,fn,tn = smp.metrics.get_stats(pred_mask.long(),target_mask.long(),mode='binary')
                #Calculate the iou
                iou_batch = tp/(tp+fp+fn)
                IoU.append(iou_batch)
                loss_list.append(torch.abs(loss))
        
        return loss_list, IoU
    
    def training(self):
        n_cpu = os.cpu_count()                
                        
        IoU_perEpoch_train = []
        loss_perEpoch_train = []
        IoU_perEpoch_val = []
        loss_perEpoch_val = []
        #train_loader = DataLoader(self.train_dataset,self.config.batch_size,shuffle=True,num_workers=n_cpu)
        #val_loader = DataLoader(self.val_dataset,len(self.val_dataset),shuffle=False,num_workers=n_cpu)
        
        for epoch in range(self.NUM_EPOCHS):
            loss_list, IoU = self.train_oneEpoch()
            #training loss and IoU over training dataset
            with torch.no_grad():
                IoU_list = [i.squeeze(dim=1).tolist() for i in IoU]
                IoU_epoch = sum(IoU_list[0])/len(IoU_list[0])
                
                IoU_perEpoch_train.append(IoU_epoch)
                
                loss_perEpoch_train.append(sum(loss_list)/len(loss_list))
                #IoU and loss over validation dataset
                for x,y in self.val_loader:
                    x = x.float().to(device=DEVICE)
                    y = y/255
                    y = y.float().to(device=DEVICE)
                    val_predictions = self.model(x)
                    
                    val_prob_mask = val_predictions.sigmoid()
                    val_pred_mask = (val_prob_mask > 0.5).float()
                    
                    tp, fp, fn, tn = smp.metrics.get_stats(val_pred_mask.long(), y.long(), mode="binary")
                    
                    iou = tp/(tp+fp+fn)
                    loss_val = torch.abs(self.loss_fn(val_predictions, y))
                IoU_perEpoch_val.append(torch.mean(iou))
                loss_perEpoch_val.append(torch.mean(loss_val))
                print(f"epoch:{epoch}, training_loss:{sum(loss_list)/len(loss_list)},validation_loss:{torch.mean(loss_val)} ,IoU_train:{IoU_epoch}, IoU_val:{torch.mean(iou)}")
        torch.save(self.model.state_dict(),'U-Net_checkpoint_epoch'+f"{epoch+1}"+'.pth')
        return IoU_perEpoch_train,IoU_perEpoch_val,loss_perEpoch_train, loss_perEpoch_val
    
    def plot_results(self,loss_perEpoch_train,loss_perEpoch_val,IoU_perEpoch_val,IoU_perEpoch_train):
        train_loss = [i.tolist() for i in loss_perEpoch_train]
        val_loss = [i.tolist() for i in loss_perEpoch_val]
        val_iou = [i.tolist() for i in IoU_perEpoch_val]
        train_iou = IoU_perEpoch_train
        x = np.arange(self.NUM_EPOCHS)
        
        print("Mean IoU over the training dataset:",train_iou[-1])
        print("Mean IoU over the validation dataset:",val_iou[-1])
        
        #Saving the loss and IOU values
        all_results={'Training_loss':train_loss,'Validation_loss':val_loss,'Training_IOU':train_iou,'Validation_IOU':val_iou}
        dataframe = pd.DataFrame(all_results)
        dataframe.to_csv('Training_Results.csv')
        
        fig,ax = plt.subplots(1,4,figsize=(12,4))
        ax[0].plot(x,train_loss)
        ax[0].set_title('training_loss')
        ax[0].set_xlabel('Epoch')
        ax[1].plot(x,val_loss)
        ax[1].set_title('validation_loss')
        ax[1].set_xlabel('Epoch')
        ax[2].plot(x,train_iou)
        ax[2].set_title('IoU_train')
        ax[2].set_xlabel('Epoch')
        ax[3].plot(x,val_iou)
        ax[3].set_title('IoU_val')
        ax[3].set_xlabel('Epoch')
        