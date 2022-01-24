from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

import os
import pandas as pd


x=os.listdir('./images')
y=os.listdir('./masks')

df=pd.DataFrame(list(zip(x,y)),columns=["image_path","mask_path"])



def TrainDataSegReader(train_dataframe,target_size_tuple,batch_size,rand_augm=True):
        if rand_augm==True:
            datagen_images=ImageDataGenerator(rescale=1./255.,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

            datagen_masks=ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest', preprocessing_function=CustomPreprocess)
        else:
            datagen_images=ImageDataGenerator(rescale=1./255.)
            datagen_masks=ImageDataGenerator()

        img_train_generator=datagen_images.flow_from_dataframe( 
            dataframe=train_dataframe, 
            directory=None,
            target_size=target_size_tuple,
            batch_size=batch_size,
            class_mode=None,
            shuffle=True, 
            x_col=train_dataframe.columns[0],                   
            y_col=train_dataframe.columns[2],                                                                      
            seed=42)
        Mask_train_generator=datagen_masks.flow_from_dataframe( 
            dataframe=train_dataframe,
             directory=None,
             target_size=target_size_tuple,
             batch_size=batch_size,
             x_col=train_dataframe.columns[1],                   
             y_col=train_dataframe.columns[2],
             class_mode=None,
             shuffle=True,                                       
             seed=42,
             color_mode="grayscale")
        
        return zip(img_train_generator, Mask_train_generator)
    






def CustomPreprocess(imageBatch):
    condition=tf.equal(imageBatch,0)
    falseValues=tf.ones(tf.shape(imageBatch))
    trueValues=tf.zeros(tf.shape(imageBatch))
    imageBatch=tf.where(condition,trueValues,falseValues)
    return imageBatch
def TestDataSegReader(train_dataframe,target_size_tuple,batch_size):
        
        datagen_images=ImageDataGenerator(rescale=1./255.,)

        img_test_generator=datagen_images.flow_from_dataframe( 
            dataframe=train_dataframe, 
            directory=None,
            target_size=target_size_tuple,
            batch_size=batch_size,
            class_mode=None,
            shuffle=True, 
            x_col=train_dataframe.columns[0],                   
            y_col=train_dataframe.columns[2],                                                                      
            seed=42)
       
        
        return img_test_generator

