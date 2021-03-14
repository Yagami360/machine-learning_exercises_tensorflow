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

    #----------------------
    # train data
    #----------------------
    # image_s (train)
    image_s_trains = np.zeros( (n_trains, image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(train_image_s_names_path):
        img = cv2.imread(name)/255
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        image_s_trains[i] = img

    # image_t (train)
    image_t_trains = np.zeros( (n_trains, image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(train_image_t_names_path):
        img = cv2.imread(name)/255
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        image_t_trains[i] = img

    #----------------------
    # valid data
    #----------------------
    # image_s (valid)
    image_s_valids = np.zeros( (n_valids, image_height, image_width, 3), dtype=np.float32 )
    for i, name in enumerate(valid_image_s_names_path):
        img = cv2.imread(name)/255
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        image_s_valids[i] = img

    # image_t (valid)
    image_t_valids = np.zeros( (n_valids, image_height, image_width, 3), dtype=np.float32 )
    for i, name in enumerate(valid_image_t_names_path):
        img = cv2.imread(name)/255
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        image_t_valids[i] = img

    return image_s_trains, image_t_trains, image_s_valids, image_t_valids


class TempleteDataGen(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, datamode =  "train", image_height = 128, image_width = 128, n_channels = 3, batch_size = 4, n_workers = -1 ):
        super(TempleteDataGen, self).__init__()
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.batch_size = batch_size

        self.image_s_dir = os.path.join( self.dataset_dir, "image_s" )
        self.image_t_dir = os.path.join( self.dataset_dir, "image_t" )
        self.image_s_names = sorted( [f for f in os.listdir(self.image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.image_t_names = sorted( [f for f in os.listdir(self.image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.image_s_names_path = [ os.path.join(self.image_s_dir, image_s_name) for image_s_name in self.image_s_names ]
        self.image_t_names_path = [ os.path.join(self.image_t_dir, image_t_name) for image_t_name in self.image_t_names ]
        return

    def __len__(self):
        """
        __len__メソッドは 1epoch あたりのイテレーション回数。
        通常は、サンプル数をバッチサイズで割った値（の切り上げ）
        ここでは、エポック数ではなく step 数で計測させるため 1 を返す
        """
        if( self.datamode == "train" ):
            return 1
        else:
            return math.ceil(len(self.image_s_names_path) / self.batch_size)

    def __getitem__(self, idx):
        #print( "idx : ", idx )
        idx_start = idx * self.batch_size
        idx_last = idx_start + self.batch_size
        image_s_names_batch = self.image_s_names_path[idx_start:idx_last]
        image_t_names_batch = self.image_t_names_path[idx_start:idx_last]
        if idx_start > len(self.image_s_names_path):
            idx_start = len(self.image_s_names_path)

        # image_s
        image_s_batch = np.zeros( (self.batch_size, self.image_height, self.image_width, self.n_channels), dtype=np.float32 )
        for i, name in enumerate(image_s_names_batch):
            img = cv2.imread(name)/255
            img = cv2.resize( img, (self.image_height, self.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
            image_s_batch[i] = img

        # image_t
        image_t_batch = np.zeros( (self.batch_size, self.image_height, self.image_width, self.n_channels), dtype=np.float32 )
        for i, name in enumerate(image_t_names_batch):
            img = cv2.imread(name)/255
            img = cv2.resize( img, (self.image_height, self.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
            image_t_batch[i] = img
        
        # 学習(fit_generatorメソッド)では説明変数と目的変数の両方、予測(predict_generatorメソッド)では説明変数のみ扱うため、それぞれ tarin と test で異なる戻り値を設定
        if( self.datamode == "train" ):
            return image_s_batch, image_t_batch
        else:
            return image_s_batch

    def on_epoch_end(self):
        """
        1エポック分の処理が完了した際に実行される。
        属性で持っている（__getitem__関数実行後も残る）データなどの破棄処理や
        コールバックなど、必要な処理があれば記載する。
        """
        return
