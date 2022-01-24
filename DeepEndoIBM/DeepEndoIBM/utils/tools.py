from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from typing import Optional, Dict
from PIL import ImageOps
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import PIL
import numpy as np

def display_mask(model, test_images, test_masks, n_samples, target_size, normalize_images = True, path_to_save = "./"):
    """Quick utility to display a model's prediction."""
    target_tuple = (target_size, target_size)
    if normalize_images == True:
        norm_factor = 255.
    else:
        norm_factor = 1.
    if n_samples<= len(test_images):
        images = test_images[0:n_samples]
        masks = test_masks[0:n_samples]
    else:
        images = test_images
        masks = test_masks
        
    for i, (img_path, mask_path) in enumerate(zip(images, masks)):
        
        loaded_image = load_img(img_path, target_size=target_tuple)
        img = np.array([img_to_array(loaded_image)/norm_factor])
        mask = load_img(mask_path,target_size=target_tuple)
        
        pred_mask = model.predict(img)
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_mask = ImageOps.autocontrast(array_to_img(pred_mask[0]))
        
        loaded_image.save(path_to_save + "/img_" + str(i) + ".png")
        mask.save(path_to_save + "/mask_gt_" + str(i) + ".png")
        pred_mask.save(path_to_save + "/mask_pred_" + str(i) + ".png")



def TrainDataSegReader(train_dataframe: pd.DataFrame, target_size_tuple: tuple, batch_size, validation: bool = False, augmentations_config: Optional[Dict] = None):
        
        
        rescale = None
        if augmentations_config is not None:
            config = augmentations_config.copy()
            if "rescale" in config.keys():
                rescale = 1 / config.pop("rescale")
        else:
            config = {}
        if validation:
            config = {}
        datagen_images=ImageDataGenerator(rescale=rescale, **config)
        datagen_masks=ImageDataGenerator(preprocessing_function=CustomPreprocess, **config)

        img_train_generator=datagen_images.flow_from_dataframe( 
            dataframe=train_dataframe, 
            directory=None,
            target_size=target_size_tuple,
            batch_size=batch_size,
            class_mode=None,
            shuffle=True, 
            x_col=train_dataframe.columns[0],                   
            seed=42)
        Mask_train_generator=datagen_masks.flow_from_dataframe( 
            dataframe=train_dataframe,
             directory=None,
             target_size=target_size_tuple,
             batch_size=batch_size,
             x_col=train_dataframe.columns[1],                   
             class_mode=None,
             shuffle=True,                                       
             seed=42,
             color_mode="grayscale")
        
        return zip(img_train_generator, Mask_train_generator)
    

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
            seed=42)
       
        
        return img_test_generator


def PreprocessLabels(labels: list):
    new_labels = []
    for label in labels:
        new_labels.append(label[0])
    return new_labels
    



def CustomPreprocess(imageBatch):
    condition=tf.equal(imageBatch,0.0)
    falseValues=tf.ones(tf.shape(imageBatch))
    trueValues=tf.zeros(tf.shape(imageBatch))
    imageBatch=tf.where(condition,trueValues,falseValues)
    return imageBatch