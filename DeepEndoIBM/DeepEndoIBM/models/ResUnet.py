# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:45:23 2021

@author: cypri
"""

import numpy as np 
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras



def stem(x, filters, kernel_size=(3, 3)):
    res = Conv2D(filters, kernel_size, padding="same", strides=1)(x)
    res =BatchNormalization()(res)
    res=Activation("relu")(res)
    res=Conv2D(filters, kernel_size, padding="same", strides=1)(res)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=1)(x)
    
    output = Add()([res, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), strides=1):
    res =BatchNormalization()(x)
    res=Activation("relu")(res)
    res=Conv2D(filters, kernel_size, padding="same", strides=strides)(res)
    res =BatchNormalization()(res)
    res=Activation("relu")(res)
    res=Conv2D(filters, kernel_size, padding="same", strides=1)(res)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
    
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip,filters):
    u = Conv2DTranspose(filters, kernel_size=(2, 2), strides=2, padding="same")(x)
    c = Concatenate()([u, xskip])
    return c



def ResUnet(input_size, num_classes):
    shape = (input_size, input_size, 3)
    f = [64, 128, 256, 128, 256]
    inputs =Input(shape=shape)
    
    ## Encoder

    e0 = stem(inputs, 64)
    e1 = residual_block(e0, 128, strides=2)
    e2 = residual_block(e1, 256, strides=2)
    
    ## Bridge
    b0=residual_block(e2, 512, strides=2)
    
    ## Decoder
    u1 = upsample_concat_block(b0, e2, 256)
    d1 = residual_block(u1, 256)
    
    u2 = upsample_concat_block(d1, e1,128)
    d2 = residual_block(u2, 128)
    
    u3 = upsample_concat_block(d2, e0,64)
    d3 = residual_block(u3, 64)
    
   
    
    outputs = Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(d3)
    model = Model(inputs, outputs)
    return model