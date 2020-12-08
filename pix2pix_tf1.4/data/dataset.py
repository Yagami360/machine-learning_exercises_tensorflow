import os
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# 自作モジュール
from utils.utils import set_random_seed, numerical_sort
from utils.utils import load_image_tsr_from_file, sava_image_tsr

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def load_dataset(
    dataset_dir, 
    image_height = 128, image_width = 128, n_channels =3,
):
    image_s_dir = os.path.join( dataset_dir, "image_s" )
    image_t_dir = os.path.join( dataset_dir, "image_t" )
    image_s_names = sorted( [f for f in os.listdir(image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_t_names = sorted( [f for f in os.listdir(image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_s_names_path = [ os.path.join(image_s_dir, image_s_name) for image_s_name in image_s_names ]
    image_s_names_path = [ os.path.join(image_t_dir, image_t_name) for image_t_name in image_t_names ]

    # tf.data.Dataset の構築
    dataset_path_s = tf.data.Dataset.from_tensor_slices(image_s_names_path)
    for item in dataset_path_s:
        print("dataset_path_s : ", item)

    # map() で関数を適用
    dataset_s = dataset_path_s.map(load_image_tsr_from_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Dataloader を用いたデータ取り出し処理
    for i, image_tsr in enumerate(dataset_s):
        #print("image_tsr : ", image_tsr)
        print("image_tsr.shape : ", image_tsr.shape)
        sava_image_tsr( image_tsr, "_debug/image_tsr_{}.png".format(i) )
        
    return


class TempleteDataset(object):
    def __init__(self, dataset_dir, image_height = 128, image_width = 128, n_channels = 3, n_workers = -1 ):
        image_s_dir = os.path.join( dataset_dir, "image_s" )
        image_t_dir = os.path.join( dataset_dir, "image_t" )
        image_s_names = sorted( [f for f in os.listdir(image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        image_t_names = sorted( [f for f in os.listdir(image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        image_s_names_path = [ os.path.join(image_s_dir, image_s_name) for image_s_name in image_s_names ]
        image_s_names_path = [ os.path.join(image_t_dir, image_t_name) for image_t_name in image_t_names ]

        # dataloader の構築
        if( n_workers == -1 ):
            dataset_s = tf.data.Dataset.from_tensor_slices(image_s_names_path).map(self.load_image_tsr_from_file, num_parallel_calls=tf.data.experimental.AUTOTUNE )
        else:
            dataset_s = tf.data.Dataset.from_tensor_slices(image_s_names_path).map(self.load_image_tsr_from_file, num_parallel_calls=n_workers )            
        return

    def load_image_tsr_from_file( self, image_path, image_height = 128, image_width = 128, normalize = True, offset = False ):
        image_data = tf.io.read_file(image_path)
        image_tsr = tf.image.decode_image(image_data)
        image_tsr = tf.image.resize(image_tsr, [image_height, image_width])
        if( normalize ):
            image_tsr = tf.cast(image_tsr, tf.float32) / 255.0
            if( offset ):
                image_tsr = image_tsr * 2.0 - 1.0

        return image_tsr
    