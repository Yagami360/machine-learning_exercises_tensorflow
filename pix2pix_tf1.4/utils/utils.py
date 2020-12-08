# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import cv2
import imageio
import random
import re

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

#====================================================
# 画像関連
#====================================================
def load_image_tsr_from_file( image_path, normalize = True, offset = False ):
    image_data = tf.io.read_file(image_path)
    image_tsr = tf.image.decode_image(image_data)
    if( normalize ):
        image_tsr = tf.cast(image_tsr, tf.float32) / 255.0
        if( offset ):
            image_tsr = image_tsr * 2.0 - 1.0

    return image_tsr

def sava_image_tsr( image_tsr, image_path, normalize = True, offset = False ):
    #print( "[image_tsr] dtype={}, min={}, max={}".format(image_tsr.dtype, np.min(image_tsr.numpy()), np.max(image_tsr.numpy())) )
    if( offset ):
        image_tsr = ( image_tsr + 1.0 ) * 0.5
    if( normalize ):
        image_tsr = tf.cast(image_tsr * 255, tf.uint8)

    image_np = image_tsr.numpy()
    #print( "[image_np] dtype={}, min={}, max={}".format(image_np.dtype, np.min(image_np), np.max(image_np)) )
    Image.fromarray(image_np).save(image_path)
    return


#====================================================
# その他
#====================================================
def set_random_seed(seed=72):
    np.random.seed(seed)
    random.seed(seed)
    if( tf.__version__.split(".")[0] == "1" ):
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)
    
    return

def numerical_sort(value):
    """
    数字が含まれているファイル名も正しくソート
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
