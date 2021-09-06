"""tfmodel.py - basic models using TensorFlow

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

class BasicConv1DModel(Model):
    def __init__(self):
        super(BasicConv1DModel, self).__init__()
        self.conv1 = layers.Convolution1D(32, 3, activation='relu')
    
    def call(self, x):
        return self.conv1(x)

class BasicConv2DModel(Model):
    def __init__(self):
        super(BasicConv2DModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

