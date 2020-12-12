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

#====================================================
# 画像関連
#====================================================
def load_image_tsr_from_file( image_path, normalize = True, offset = True ):
    image_data = tf.io.read_file(image_path)
    image_tsr = tf.image.decode_image(image_data, expand_animations = False)    # gif 以外読み込み
    if( normalize ):
        image_tsr = tf.cast(image_tsr, tf.float32) / 255.0
        if( offset ):
            image_tsr = image_tsr * 2.0 - 1.0

    return image_tsr

def sava_image_tsr( image_tsr, image_path, normalize = True, offset = True ):
    #print( "[image_tsr] dtype={}, min={}, max={}".format(image_tsr.dtype, np.min(image_tsr.numpy()), np.max(image_tsr.numpy())) )
    if( offset ):
        image_tsr = ( image_tsr + 1.0 ) * 0.5
    if( normalize ):
        image_tsr = tf.cast(image_tsr * 255, tf.uint8)

    image_np = image_tsr.numpy()
    #print( "[image_np] dtype={}, min={}, max={}".format(image_np.dtype, np.min(image_np), np.max(image_np)) )
    Image.fromarray(image_np).save(image_path)
    return

def resize_image_tsr( image_tsr, image_height, image_width, method = tf.image.ResizeMethod.BILINEAR ):
    """
    [args]
        method
            ResizeMethod.BILINEAR, ResizeMethod.NEAREST_NEIGHBOR, ResizeMethod.BICUBIC, ResizeMethod.AREA
    """
    image_tsr = tf.image.resize( image_tsr, [image_height, image_width], method )
    return image_tsr

#====================================================
# TensorBoard への出力関連
#====================================================
def tensor_for_board(img_tensor, normalize = True, offset = True):
    # map into [0,1]
    if( normalize ):
        if( offset ):
            tensor = (tf.identity(img_tensor)+1) * 0.5
        else:
            tensor = tf.identity(img_tensor)
    else:
        tensor = tf.identity(img_tensor) / 255

    with tf.device("CPU:0"):
        tensor = tf.clip_by_value(tensor, 0, 1)

    if tensor.shape[1] == 1:
        tensor = tf.repeat( tensor, (1,3,1,1) )

    return tensor

def tensor_list_for_board(img_tensors_list, normalize = True, offset = True):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)
    
    batch_size, height, width, channel = tensor_for_board(img_tensors_list[0][0], normalize, offset).shape
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = tf.fill( (batch_size, canvas_h, canvas_w, channel), 0.5).numpy()
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor, normalize, offset)
            canvas[:, offset_h : offset_h + height, offset_w : offset_w + width, :] = tensor.numpy()

    canvas = tf.convert_to_tensor(canvas, dtype=tf.float32)
    return canvas

def board_add_image(board, tag_name, img_tensor, step_count, n_max_images = 32, description = None, normalize = True, offset = True ):
    tensor = tensor_for_board(img_tensor, normalize, offset)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        tf.summary.image('%s/%03d' % (tag_name, i), img, step=step_count, description=description )

    return

def board_add_images(board, tag_name, img_tensors_list, step_count, n_max_images = 32, description = None, normalize = True, offset = True ):
    tensor = tensor_list_for_board(img_tensors_list, normalize, offset)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        tf.summary.image('%s/%03d' % (tag_name, i), tf.expand_dims(img, 0), step=step_count, description=description )

    return

#====================================================
# TFRecord 関数
#====================================================
def write_tfrecord_from_file( load_file_names, save_file_name = "images.tfrec" ):
    dataset = tf.data.Dataset.from_tensor_slices(load_file_names).map(tf.io.read_file)
    writer = tf.data.experimental.TFRecordWriter(save_file_name)
    writer.write(dataset)
    return

def write_tfrecord_from_dataset( dataset, save_file_name = "images.tfrec" ):
    dataset.map(tf.io.read_file)
    writer = tf.data.experimental.TFRecordWriter(save_file_name)
    writer.write(dataset)
    return

def load_tfrecord_from_file( file_names ):
    dataset = tf.data.TFRecordDataset(file_names)
    return dataset

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
