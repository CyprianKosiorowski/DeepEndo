# -*- coding: utf-8 -*-


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import scipy
from scipy import ndimage 
import random
from typing import Optional

from skimage.transform import resize
class LocalizationGenerator(keras.utils.Sequence):
    def __init__(self, batch_size: int, img_size: int, input_img_paths: list, target_img_paths: list, labels = None, train_classifier: bool = False,random_augmentation: bool = False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.labels = labels
        self.random_augmentation=random_augmentation
        self.train_classifier = train_classifier
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def __random_augmentation(image,target):
        angles=[-20,-10,-5,5,10,20]
        angle=random.choice(angles)
        image=ndimage.rotate(image,angle,reshape=False)
        target=ndimage.rotate(target,angle,reshape=False)
        return image, target
    def __getitem__(self,idx):
        i=idx*self.batch_size
        if self.train_classifier:
            batch_labels =self.labels[i:i+self.batch_size]
        bach_input_img_paths=self.input_img_paths[i:i+self.batch_size]
        batch_input_target_img_paths=self.target_img_paths[i:i+self.batch_size]
        x=np.zeros((self.batch_size,)+self.img_size+(3,),dtype="float32")
        coordinates=[]
        for j, (path_img, path_tar) in enumerate(zip(bach_input_img_paths,batch_input_target_img_paths)):
            img=img_to_array(load_img(path_img))
            target=img_to_array(load_img(path_tar,color_mode="grayscale"))
            if self.random_augmentation==True:
                img,target=self.__random_augmentation(img,target)
            img = resize(img, (self.img_size, self.img_size))
            
            x[j]=img
           # y[j]=np.expand_dims(target,2)
            x=np.where(np.any(target==255, axis=0))
            y=np.where(np.any(target==255, axis=1))
            x0=min(x[0])/self.img_size
            y0=min(y[0])/self.img_size
            x1=max(x[0])/self.img_size
            y1=max(y[0])/self.img_size
          #  target = resize(target, (self.img_size, self.img_size))
            coordinates.append([x0,y0,x1,y1])
        coordinates = np.array(coordinates)
        
        if self.train_classifier:
            y=[coordinates, batch_labels]
        else:
            y= coordinates
        return x, y