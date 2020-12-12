import os
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split

# TensorFlow ライブラリ
import tensorflow as tf

# 自作モジュール
from utils.utils import set_random_seed, numerical_sort
from utils.utils import load_image_tsr_from_file, sava_image_tsr, resize_image_tsr
from utils.utils import write_tfrecord_from_file, write_tfrecord_from_dataset, load_tfrecord_from_file

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

#@tf.function
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
    val_rate = 0.01,
    use_tfrecord = True, use_prefeatch = True,
    seed = 71,
):
    image_s_dir = os.path.join( dataset_dir, "image_s" )
    image_t_dir = os.path.join( dataset_dir, "image_t" )
    image_s_names = sorted( [f for f in os.listdir(image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_t_names = sorted( [f for f in os.listdir(image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
    image_s_names_path = [ os.path.join(image_s_dir, image_s_name) for image_s_name in image_s_names ]
    image_t_names_path = [ os.path.join(image_t_dir, image_t_name) for image_t_name in image_t_names ]

    # split train and valid
    train_image_s_names_path, valid_image_s_names_path, train_image_t_names_path, valid_image_t_names_path = train_test_split( image_s_names_path, image_t_names_path, test_size=val_rate, random_state=seed )
    n_trains = len(train_image_s_names_path)
    n_valids = len(valid_image_s_names_path)

    # tfrecord
    if( use_tfrecord ):
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord")) ):
            os.mkdir(os.path.join(dataset_dir, "tfrecord"))
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "train")) ):
            os.mkdir(os.path.join(dataset_dir, "tfrecord", "train"))
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "valid")) ):
            os.mkdir(os.path.join(dataset_dir, "tfrecord", "valid"))

        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "train", "image_s.tfrec")) ):
            write_tfrecord_from_file( train_image_s_names_path, os.path.join( dataset_dir, "tfrecord", "train", "image_s.tfrec") )
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "valid", "image_s.tfrec")) ):
            write_tfrecord_from_file( valid_image_s_names_path, os.path.join( dataset_dir, "tfrecord", "valid", "image_s.tfrec") )
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "train", "image_t.tfrec")) ):
            write_tfrecord_from_file( train_image_s_names_path, os.path.join( dataset_dir, "tfrecord", "train", "image_t.tfrec") )
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "valid", "image_t.tfrec")) ):
            write_tfrecord_from_file( train_image_s_names_path, os.path.join( dataset_dir, "tfrecord", "valid", "image_t.tfrec") )

        ds_train_s = tf.data.TFRecordDataset( os.path.join(dataset_dir, "tfrecord", "train", "image_s.tfrec") )
        ds_valid_s = tf.data.TFRecordDataset( os.path.join(dataset_dir, "tfrecord", "valid", "image_s.tfrec") )
        ds_train_t = tf.data.TFRecordDataset( os.path.join(dataset_dir, "tfrecord", "train", "image_t.tfrec") )
        ds_valid_t = tf.data.TFRecordDataset( os.path.join(dataset_dir, "tfrecord", "valid", "image_t.tfrec") )
    else:
        # tf.data.Dataset の構築
        ds_train_s = tf.data.Dataset.from_tensor_slices(train_image_s_names_path)
        ds_valid_s = tf.data.Dataset.from_tensor_slices(valid_image_s_names_path)
        ds_train_t = tf.data.Dataset.from_tensor_slices(train_image_t_names_path)
        ds_valid_t = tf.data.Dataset.from_tensor_slices(valid_image_t_names_path)

    # from_tensor_slices() で指定した値に対して、map() で関数を適用して変換
    ds_train_s = ds_train_s.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_valid_s = ds_valid_s.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train_t = ds_train_t.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_valid_t = ds_valid_t.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # zip 化して image_t, image_s のペアデータにする
    ds_train = tf.data.Dataset.zip((ds_train_s, ds_train_t)).shuffle(n_trains).batch(batch_size)
    ds_valid = tf.data.Dataset.zip((ds_valid_s, ds_valid_t)).batch(1)
    if( use_prefeatch ):
        ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds_valid = ds_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds_train, ds_valid, n_trains, n_valids


"""
class TempleteDataset(object):
    # [ToDO] PyTorch のデータローダのような挙動をするデータローダ
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

    #@tf.function
    def preprocessing( self, image_path, image_height = 128, image_width = 128, normalize = True, offset = False ):
        image_tsr = load_image_tsr_from_file( image_path, normalize = True, offset = False )
        image_tsr = resize_image_tsr( image_tsr, image_height, image_width )
        return image_tsr
    
    def get_next( self, index ):
        return
"""