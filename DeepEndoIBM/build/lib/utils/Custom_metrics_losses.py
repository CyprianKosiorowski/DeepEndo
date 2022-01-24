# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:53:03 2021

@author: cypri
"""
from tensorflow.keras.metrics import MeanIoU,Recall, Precision

import tensorflow as tf

class MyMeanIOU(MeanIoU):
    def __init__(self, name='MyMeanIOU', **kwargs):
        super(MyMeanIOU, self).__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

class MyPrecission(Precision):
    def __init__(self, name='MyPrecission', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

class MyRecall(Recall):
    def __init__(self, name='MyRecall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred=tf.cast(tf.argmax(y_pred, axis=-1),tf.float32)
    y_pred = tf.keras.layers.Flatten()(y_pred )
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


class DiceCoeff(tf.keras.metrics.Metric):

  def __init__(self, name='DiceCoeff', **kwargs):
    super(DiceCoeff, self).__init__(name=name, **kwargs)
    self.dice = self.add_weight(name='dice', initializer='zeros')
  def update_state(self, y_true, y_pred, sample_weight=None):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred=tf.cast(tf.argmax(y_pred, axis=-1),tf.float32)
    y_pred = tf.keras.layers.Flatten()(y_pred )
    intersection = tf.reduce_sum(y_true * y_pred)
    self.dice=(2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
  def result(self):
    return self.dice