# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 01:31:14 2021

@author: cypri
"""
from DeepEndoIBM.config.config import LocalizationConfig, SegmentationConfig

from DeepEndoIBM.models.DeepLabv3 import Deeplabv3
from DeepEndoIBM.models.DeepLabv3plus import DeeplabV3Plus
from DeepEndoIBM.models.Unet import Unet
from DeepEndoIBM.models.ResUnet import ResUnet
from DeepEndoIBM.models.ResUnetPlusPlus import ResUnetPlusPlus
from DeepEndoIBM.models.DoubleUnet import DoubleUnet
from DeepEndoIBM.models.TriUnet import TriUnet
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
MODELS = {
 "Unet": Unet,
 "DeeplabV3Plus": DeeplabV3Plus,
 "ResUnet": ResUnet,
 "ResUnetPlusPlus": ResUnetPlusPlus,
 "DoubleUnet": DoubleUnet,
 "TriUnet": TriUnet, 
 }



def GetSegmentationModel(config: SegmentationConfig):
    return MODELS[config.model_name](**config.model_config)