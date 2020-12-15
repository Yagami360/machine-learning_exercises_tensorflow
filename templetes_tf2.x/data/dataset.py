import os
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io

# sklearn
from sklearn.model_selection import train_test_split

# TensorFlow ライブラリ
import tensorflow as tf

# 自作モジュール
from utils.utils import *


IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

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

        # tfrecord の書き込み
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "train.tfrecord")) ):
            write_tfrecord_from_files( train_image_s_names_path, train_image_t_names_path, os.path.join(dataset_dir, "tfrecord", "train.tfrecord") )
        if not( os.path.exists(os.path.join(dataset_dir, "tfrecord", "valid.tfrecord")) ):
            write_tfrecord_from_files( valid_image_s_names_path, valid_image_t_names_path, os.path.join(dataset_dir, "tfrecord", "valid.tfrecord") )

        # tfrecord の読み込み
        ds_train = load_tfrecord_from_file_tf2( os.path.join(dataset_dir, "tfrecord", "train.tfrecord") ).shuffle(n_trains).batch(batch_size)
        ds_valid = load_tfrecord_from_file_tf2( os.path.join(dataset_dir, "tfrecord", "valid.tfrecord") ).batch(1)
        if( use_prefeatch ):
            ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            ds_valid = ds_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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


def write_tfrecord_from_files( image_s_names, image_t_names, tfrecord_name, image_height = 128, image_width = 128 ):
    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for image_s_name, image_t_name in zip(image_s_names, image_t_names):
            # 画像を読み込み
            image_s_pillow = Image.open(image_s_name).convert("RGB").resize((image_width, image_height))
            image_t_pillow = Image.open(image_t_name).convert("RGB").resize((image_width, image_height))

            # 画像は byte 型に変換
            bytes_io = io.BytesIO()
            image_s_pillow.save(bytes_io, format='PNG')
            image_s_byte = bytes_io.getvalue()

            bytes_io = io.BytesIO()
            image_t_pillow.save(bytes_io, format='PNG')
            image_t_byte = bytes_io.getvalue()

            # tf.Example の作成 / TFRecordは、tf.train.Example を１つのレコードの単位として書き込む。
            example = tf.train.Example(
                # 書き込み特徴量 / tf.train.Example は数値や画像などの固定長のリストを扱い、各レコードの値を tf.train.Feature で指定する
                features = tf.train.Features(
                    feature = {
                        "image_s" : tf.train.Feature( bytes_list=tf.train.BytesList(value=[image_s_byte]) ),
                        "image_t" : tf.train.Feature( bytes_list=tf.train.BytesList(value=[image_t_byte]) ),
                        "image_height" : tf.train.Feature( int64_list=tf.train.Int64List(value=[image_height]) ),
                        "image_width" : tf.train.Feature( int64_list=tf.train.Int64List(value=[image_width]) ),
                    }
                )
            )

            # tf.Example を tfrecord ファイルに書き込む
            writer.write( example.SerializeToString() )

    return

def load_tfrecord_from_file_tf1( tfrecord_name ):
    """
    tensorflow 1.x 系での tfrecord 読み込み処理
    """
    reader = tf.TFRecordReader()

    # tfrecord ファイルを queue 形式で読み込む
    queue = tf.train.string_input_producer([tfrecord_name])

    # queue 内の TFRecords ファイルを読み込み、デシリアライズする
    _, serialized_example = reader.read(queue)

    # tf.parse_single_example を使用して tfrecord を読み込む
    features = tf.parse_single_example(
        serialized_example,
        # 読み込み特徴量のディクショナリ    
        features = {
            "image_s" : tf.io.FixedLenFeature([], tf.string),
            "image_t" : tf.io.FixedLenFeature([], tf.string),
            "image_height" : tf.io.FixedLenFeature([], tf.int64),
            "image_width" : tf.io.FixedLenFeature([], tf.int64),
        }
    )

    # string 型を変換
    image_s_tsr = tf.decode_raw(features['image_s'], tf.float32)
    image_t_tsr = tf.decode_raw(features['image_t'], tf.float32)

    # データローダに変換
    dataset_s = tf.data.Dataset.from_tensor_slices(image_s_tsr)
    dataset_t = tf.data.Dataset.from_tensor_slices(image_t_tsr)
    dataset = tf.data.Dataset.zip((dataset_s, dataset_t))
    return dataset

def load_tfrecord_from_file_tf2( tfrecord_name, display_tfrecord = False ):
    def parse_example(example_proto):
        # tf.parse_single_example を使用して tfrecord を読み込む
        features = tf.io.parse_single_example(
            example_proto,
            # 読み込み特徴量のディクショナリ    
            features = {
                "image_s" : tf.io.FixedLenFeature([], tf.string),
                "image_t" : tf.io.FixedLenFeature([], tf.string),
                "image_height" : tf.io.FixedLenFeature([], tf.int64),
                "image_width" : tf.io.FixedLenFeature([], tf.int64),
            }
        )
        return features

    # tfrecord ファイルの中身を表示
    if( display_tfrecord ):
        for example in tf.compat.v1.io.tf_record_iterator(tfrecord_name): 
            tfrecord_str = tf.train.Example.FromString(example)
            print( "tfrecord_str : \n", tfrecord_str )

    # map() で tf.parse_single_example を使用して tfrecord を読み込み処理 
    dataset = tf.data.TFRecordDataset(tfrecord_name).map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 読み込んだ tfrecord ファイルのデータに対して前処理を適用 ＆ データローダーの形式変換
    image_s_tsr_list = []
    image_t_tsr_list = []
    for features in dataset:
        image_s_tsr = tf.io.decode_png(features['image_s'], dtype = tf.uint8)
        image_t_tsr = tf.io.decode_png(features['image_t'], dtype = tf.uint8)
        image_s_tsr = tf.cast(image_s_tsr, tf.float32 )
        image_t_tsr = tf.cast(image_t_tsr, tf.float32 )
        image_s_tsr = ( image_s_tsr / 255 ) * 2 - 1.0
        image_t_tsr = ( image_t_tsr / 255 ) * 2 - 1.0
        image_s_tsr_list.append( image_s_tsr )
        image_t_tsr_list.append( image_t_tsr )

    dataset_s = tf.data.Dataset.from_tensor_slices(image_s_tsr_list)
    dataset_t = tf.data.Dataset.from_tensor_slices(image_t_tsr_list)
    dataset = tf.data.Dataset.zip((dataset_s, dataset_t))
    return dataset

#@tf.function
def preprocessing( image_path, image_height = 128, image_width = 128 ):
    """
    前処理スクリプト
    """
    image_tsr = load_image_tsr_from_file(image_path)
    image_tsr = resize_image_tsr( image_tsr, image_height, image_width )
    return image_tsr


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