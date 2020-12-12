import os
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# TensorFlow ライブラリ
import tensorflow as tf

# 自作モジュール
from utils.utils import set_random_seed, numerical_sort
from utils.utils import load_image_tsr_from_file, sava_image_tsr, resize_image_tsr

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def preprocessing( image_path, image_height = 128, image_width = 128 ):
    """
    前処理スクリプト
    """
    image_tsr = load_image_tsr_from_file(image_path)
    image_tsr = resize_image_tsr( image_tsr, image_height, image_width )
    return image_tsr

def load_dataset(
    dataset_dir, 
    image_height = 128, image_width = 128, n_channels =3, batch_size = 4,
    use_prefeatch = True,
):
    image_s_dir = os.path.join( dataset_dir, "image_s" )
    image_t_dir = os.path.join( dataset_dir, "image_t" )
    image_s_names = sorted( [f for f in os.listdir(image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_t_names = sorted( [f for f in os.listdir(image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_s_names_path = [ os.path.join(image_s_dir, image_s_name) for image_s_name in image_s_names ]
    image_t_names_path = [ os.path.join(image_t_dir, image_t_name) for image_t_name in image_t_names ]

    # tf.data.Dataset の構築
    dataset_s = tf.data.Dataset.from_tensor_slices(image_s_names_path)
    dataset_t = tf.data.Dataset.from_tensor_slices(image_t_names_path)
    """
    for item in dataset_s:
        print("dataset_s : ", item)
    """
    
    # from_tensor_slices() で指定した値に対して、map() で関数を適用して変換
    dataset_s = dataset_s.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_t = dataset_t.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # zip 化して image_t, image_s のペアデータにする
    dataset = tf.data.Dataset.zip((dataset_s,dataset_t)).shuffle(len(image_s_names_path)).batch(batch_size)
    if( use_prefeatch ):
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class TempleteDataset(object):
    def __init__(self, dataset_dir, image_height = 128, image_width = 128, n_channels = 3, n_workers = -1 ):
        image_s_dir = os.path.join( dataset_dir, "image_s" )
        image_t_dir = os.path.join( dataset_dir, "image_t" )
        image_s_names = sorted( [f for f in os.listdir(image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        image_t_names = sorted( [f for f in os.listdir(image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        image_s_names_path = [ os.path.join(image_s_dir, image_s_name) for image_s_name in image_s_names ]
        image_t_names_path = [ os.path.join(image_t_dir, image_t_name) for image_t_name in image_t_names ]

        # dataloader の構築
        if( n_workers == -1 ):
            dataset_s = tf.data.Dataset.from_tensor_slices(image_s_names_path).map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE )
            dataset_t = tf.data.Dataset.from_tensor_slices(image_t_names_path).map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE )
            self.dataset = tf.data.Dataset.zip((dataset_s,dataset_t))
        else:
            dataset_s = tf.data.Dataset.from_tensor_slices(image_s_names_path).map(self.preprocessing, num_parallel_calls=n_workers )            
            dataset_t = tf.data.Dataset.from_tensor_slices(image_t_names_path).map(self.preprocessing, num_parallel_calls=n_workers )            
            self.dataset = tf.data.Dataset.zip((dataset_s,dataset_t))

        return

    def preprocessing( self, image_path, image_height = 128, image_width = 128, normalize = True, offset = False ):
        image_tsr = load_image_tsr_from_file( image_path, normalize = True, offset = False )
        image_tsr = resize_image_tsr( image_tsr, image_height, image_width )
        return image_tsr
    
    def get_minibatch( self, index ):
        return
