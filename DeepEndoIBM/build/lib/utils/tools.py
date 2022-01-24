from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

def get_train_test_val_ers(ers_csv_path :str, train_test :float=0.2, train_val :float=0.125, mapper :dict=None, sample :bool=False) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Descripiton:
            Splitting **ERS** dataset into subsets: train, test, validation

        ers_csv_path : (str) 
            path to ERS *.csv file
        
        train_test : (float) 0.2 
            train : test splitting ratio
        
        train_val : (float) 0.125 
            train:val splitting ratio

        Return : 
            3 pandas dataframes of structure like below

        | idx | abs_path_image |    label   |
        |-----|----------------|------------|
        | xxx | /path/to/image | label name |
        | ... | .............. | .......... |
        | ... | .............. | .......... |
    """
    ers = pd.read_csv(ers_csv_path, sep=';')
    if sample:
        ers = ers.sample(n=1500)
    ers['cat'] = ers['label'].str[0]
    if not mapper:
        mapper = {
            'g':'gastro',
            'c':'colono',
            'h':'healthy',
            'b':'blood',
            'q':'quality'
        }
    else: mapper = mapper
    ers['label'] = ers['cat'].map(mapper)
    labels = ers['label']
    ers.drop(['label','cat'], axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(ers[['abs_path_image']], labels, test_size=train_test, random_state=0,stratify=labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=train_val, random_state=0,stratify=y_train)

    train_output = pd.concat([x_train, y_train], axis=1)
    test_output = pd.concat([x_test, y_test], axis=1)
    validation_output = pd.concat([x_val, y_val], axis=1)

    return [train_output, test_output, validation_output]

def get_image_mask_ers(ers_csv_path :str, train_test :float=0.1, train_val :float=0.1, mapper :dict=None, sample :bool=False) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Descripiton:
            Splitting **ERS** dataset into subsets: train, test, validation

        ers_csv_path : (str) 
            path to ERS *.csv file
        
        train_test : (float) 0.2 
            train : test splitting ratio
        
        train_val : (float) 0.125 
            train:val splitting ratio

        Return : 
            3 pandas dataframes of structure like below

        | idx | abs_path_image |    label   |
        |-----|----------------|------------|
        | xxx | /path/to/image | label name |
        | ... | .............. | .......... |
        | ... | .............. | .......... |
    """
    ers = pd.read_csv(ers_csv_path, sep=';')
    if sample:
        ers = ers.sample(n=1500)
    ers.dropna(axis=0,inplace=True)
    ers['cat'] = ers['label'].str[0]
    if not mapper:
        mapper = {
            'g':'gastro',
            'c':'colono',
            'h':'healthy',
            'b':'blood',
            'q':'quality'
        }
    else: mapper = mapper
    ers['label'] = ers['cat'].map(mapper)
    labels = ers[['abs_path_mask','label']]
    ers.drop(['label','cat'], axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(ers[['abs_path_image']], labels, test_size=train_test, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=train_val, random_state=0)

    train_output = pd.concat([x_train, y_train], axis=1)
    test_output = pd.concat([x_test, y_test], axis=1)
    validation_output = pd.concat([x_val, y_val], axis=1)

    return [train_output, test_output, validation_output]



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
            datagen_masks=ImageDataGenerator(preprocessing_function=CustomPreprocess)

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
    





def CustomPreprocess(imageBatch):
    condition=tf.equal(imageBatch,0.0)
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
            seed=42)
       
        
        return img_test_generator

def TrainDataReader(train_dataframe, target_size_tuple):
    datagen=ImageDataGenerator(rescale=1./255.)         
    train_generator=datagen.flow_from_dataframe(        
    dataframe=train_dataframe,                          
    directory="",                                       
    x_col=train_dataframe.columns[0],                   
    y_col=train_dataframe.columns[1],                   
    subset="training",                                  
    batch_size=BATCH_SIZE,                              
    seed=42,                                            
    shuffle=True,                                       
    class_mode="categorical",                           
    target_size=target_size_tuple)                      
                                                        
    return train_generator                              


def ValidDataReader(valid_dataframe, target_size_tuple):
    datagen=ImageDataGenerator(rescale=1./255.)         
    valid_generator=datagen.flow_from_dataframe(        
    dataframe=valid_dataframe,                          
    directory="",                                       
    x_col=valid_dataframe.columns[0],                   
    y_col=valid_dataframe.columns[1],                   
    batch_size=BATCH_SIZE,                              
    seed=42,                                            
    shuffle=True,                                       
    class_mode="categorical",                           
    target_size=target_size_tuple)                      
                                                        
    return valid_generator                              


def TestDataReader(test_dataframe, target_size_tuple):  
    test_datagen=ImageDataGenerator(rescale=1./255.)    
    test_generator=test_datagen.flow_from_dataframe(    
    dataframe=test_dataframe,                           
    directory="",                                       
    x_col=test_dataframe.columns[0],                    
    y_col=None,                                         
    batch_size=BATCH_SIZE,                              
    seed=42,                                            
    shuffle=False,                                      
    class_mode=None,                                    
    target_size=target_size_tuple)                      
                                                        
    return test_generator                               

