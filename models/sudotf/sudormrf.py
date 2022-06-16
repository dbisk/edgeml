"""sudormrf.py

SuDo-RM-RF model, implemented in Keras/TensorFlow.

@author Efthymios Tzinis and Dean Biskup
@email <etzinis2@illinois.edu>, <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import math

import tensorflow as tf
from tensorflow import keras


class ConvNormAct(keras.layers.Layer):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation.
    """
    
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = keras.layers.Conv1D(
            nOut, kSize, strides=stride, padding='same', use_bias=True,
            groups=groups, 
        )
        self.norm = keras.layers.GroupNorm

# TODO

