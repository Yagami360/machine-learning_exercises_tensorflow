# -*- coding:utf-8 -*-
import os
import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization
from tensorflow.keras import Model

class TempleteNetworks(Model):
    """
    ダミー用の何もしないネットワーク
    """
    def __init__( self, out_dim = 1 ):
        super(TempleteNetworks, self).__init__()
        self.dummmy_layer = Conv2D( kernel_size=1, filters=out_dim, strides=1 )
        return

    def call( self, input ):
        output = self.dummmy_layer(input)
        return output
