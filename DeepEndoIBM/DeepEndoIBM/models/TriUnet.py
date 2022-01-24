# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:04:39 2021

@author: cypri
"""
from DeepEndoIBM.models.Unet import Unet
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def TriUnet(input_size, num_classes):
    input_shape = (input_size, input_size, 3)
    input_layer = tf.keras.Input(shape=input_shape)
    unet_1 =  Unet(input_size, num_classes)
    unet_1._name = "unet_1"
    new_output_1=unet_1(input_layer)
    unet_2 =  Unet(input_size, num_classes)
    unet_2._name= "unet_2"
    new_output_2=unet_2(input_layer)
    intermidiate_output = concatenate([new_output_1, new_output_2], axis=-1)
    unet_3=Unet((input_size, input_size, num_classes*2), num_classes)
    unet_3._name = "unet_3"
    new_output_3 = unet_3(intermidiate_output)
    model = Model(input_layer, new_output_3)
    return model

    
    
    
    
    
    
    
    