# -*- coding:utf-8 -*-
import os
import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization
from tensorflow.keras import Model

class TempleteNetworks(Model):
    """
    ダミー用のネットワーク
    """
    def __init__( self, out_dim = 1 ):
        super(TempleteNetworks, self).__init__()
        self.conv = Conv2D( kernel_size=1, filters=out_dim, strides=1 )
        self.batch_norm = BatchNormalization()
        self.activate = ReLU()
        return

    def call( self, input ):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activate(output)
        return output
