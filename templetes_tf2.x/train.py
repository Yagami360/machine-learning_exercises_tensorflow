import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import cv2

# TensorFlow ライブラリ
import tensorflow as tf

# 自作モジュール
from data.dataset import load_dataset
from models.networks import TempleteNetworks
from utils.utils import set_random_seed, numerical_sort
from utils.utils import sava_image_tsr
from utils.utils import board_add_image, board_add_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=128, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=128, help="入力画像の幅（pixel単位）")
    parser.add_argument('--lr', type=float, default=0.0002, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--detect_nan', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))
        print( "tensoflow version : ", tf.__version__)

    # 出力フォルダの作成
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )

    # 実行 Device の設定
    if( tf.config.experimental.list_physical_devices("GPU") ):
        # GPU
        pass
    else:
        # CPU
        pass

    # seed 値の固定
    set_random_seed(args.seed)

    # tensorboard 出力
    board_train = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_valid = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )
    board_train.set_as_default()
    #board_valid.set_as_default()

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train = load_dataset( args.dataset_dir, image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = args.batch_size )

    #================================
    # モデルの構造を定義する。
    #================================
    model_G = TempleteNetworks()
    if( args.debug ):
        model_G( tf.zeros([args.batch_size, args.image_height, args.image_width, 3], dtype=tf.float32) )    # 動的作成されるネットワークなので、一度ネットワークに入力データを供給しないと summary() を出力できない
        model_G.summary()
    
    #================================
    # optimizer の設定
    #================================
    optimizer_G = tf.keras.optimizers.Adam( learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2 )

    #================================
    # loss 関数の設定
    #================================
    loss_fn = tf.keras.losses.MeanSquaredError()

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_prints = 1
    step = 0
    iters = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        # ミニバッチデータの取り出し
        for image_s, image_t in ds_train:
            if( args.debug and n_prints > 0 ):
                print("[image_s] shape={}, dtype={}, min={}, max={}".format(image_s.shape, image_s.dtype, np.min(image_s.numpy()), np.max(image_s.numpy())))
                print("[image_t] shape={}, dtype={}, min={}, max={}".format(image_t.shape, image_t.dtype, np.min(image_t.numpy()), np.max(image_t.numpy())))
                sava_image_tsr( image_s[0], "_debug/image_s.png" )
                sava_image_tsr( image_t[0], "_debug/image_t.png" )

            #----------------------------------------------------
            # 生成器 の forword 処理
            #----------------------------------------------------
            pass
    
            #----------------------------------------------------
            # 生成器の更新処理
            #----------------------------------------------------
            loss_G = tf.zeros([1], dtype=tf.float32)[0]

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # lr
                pass

                # loss
                with board_train.as_default():
                    tf.summary.scalar("G/loss_G", loss_G, step=step+1, description="生成器の全loss")

                # visual images
                with board_train.as_default():
                    tf.summary.image( "train/image_s", image_s, step=step+1 )
                    tf.summary.image( "train/image_t", image_t, step=step+1 )

                    visuals = [
                        [ image_s, image_t, ],
                    ]
                    board_add_images(board_train, 'train', visuals, step+1, offset = False )

            #====================================================
            # valid データでの処理
            #====================================================
            if( step != 0 and ( step % args.n_display_valid_step == 0 ) ):
                pass

            step += 1
            iters += args.batch_size
            n_prints -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            pass

    print("Finished Training Loop.")
