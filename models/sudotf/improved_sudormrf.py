"""
improved_sudormrf.py - Improved SuDO-RM-RF model, implemented in TensorFlow.
Original model implemented in PyTorch found at:
https://github.com/etzinis/sudo_rm_rf.

@author Efthymios Tzinis and Dean Biskup
@email <etzinis2@illinois.edu>, <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import math

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model


class _LayerNorm(Model):
    """Layer Normalization base class"""


