# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:33:22 2021

@author: cypri
"""
from tensorflow.keras import layers
from tensorflow import keras
def get_model(img_size, num_classes):
    inputs=keras.Input(shape=img_size+(3,))
    
    
    x=layers.Conv2D(32,3,strides=2,padding="same")(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    prev_block=x
    
    for filters in [64,128,256]:
        x=layers.Activation("relu")(x)
        x=layers.SeparableConv2D(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)
        
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        residual=layers.Conv2D(filters,1,strides=2,padding="same")(prev_block)
        x=layers.add([x,residual])
        prev_block=x
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(prev_block)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        prev_block = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model