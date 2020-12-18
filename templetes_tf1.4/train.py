import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import cv2
import warnings
warnings.simplefilter('ignore')

# TensorFlow ライブラリ
import tensorflow as tf

# 自作モジュール
from data.dataset import Dataset
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
    parser.add_argument('--use_tfrecord', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
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

    # 計算グラフ初期化
    tf.reset_default_graph()

    # Session 開始
    sess = tf.Session()

    # 実行 Device の設定
    pass

    # seed 値の固定
    set_random_seed(args.seed)

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train = Dataset( args.dataset_dir, image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = args.batch_size, use_tfrecord = args.use_tfrecord )

    #================================
    # 変数とプレースホルダを設定
    #================================
    image_s_holder = tf.placeholder(tf.float32, [args.batch_size, args.image_height, args.image_width, 1], name = "image_s_holder" )
    image_t_holder = tf.placeholder(tf.float32, [args.batch_size, args.image_height, args.image_width, 3], name = "image_t_holder")
    #image_s_holder = tf.placeholder(tf.float32, [None, args.image_height, args.image_width, 1] )
    #image_t_holder = tf.placeholder(tf.float32, [None, args.image_height, args.image_width, 3] )

    #================================
    # モデルの構造を定義する。
    #================================
    model_G = TempleteNetworks(out_dim=3)
    output_op = model_G(image_s_holder)

    #================================
    # loss 関数の設定
    #================================
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean( tf.abs(image_t_holder - output_op) )

    #================================
    # optimizer の設定
    #================================
    with tf.name_scope('optimizer'):
        optimizer_op = tf.train.AdamOptimizer( learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2 )
        train_op = optimizer_op.minimize(loss_op)

    #================================
    # tensorboard 出力
    #================================
    if( int(tf.__version__.split(".")[0]) >= 2 ):
        board_train = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name) )
        board_valid = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )
        board_train.set_as_default()
        #board_valid.set_as_default()
    else:
        board_train = tf.summary.FileWriter( os.path.join(args.tensorboard_dir, args.exper_name), sess.graph )
        board_valid = tf.summary.FileWriter( os.path.join(args.tensorboard_dir, args.exper_name + "_valid"), sess.graph )
        tf.summary.scalar("G/loss_G", loss_op)
        board_merge_op = tf.summary.merge_all()

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_prints = 1
    step = 0
    iters = 0

    # 変数初期化
    sess.run( tf.global_variables_initializer() )

    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        # データセット初期化
        sess.run(ds_train.init_iter_op)
        while True:
            try:
                # ミニバッチデータの取り出し
                image_s, image_t = sess.run(ds_train.batch_op)
                if( args.debug and n_prints > 0 ):
                    print("[image_s] shape={}, dtype={}, min={}, max={}".format(image_s.shape, image_s.dtype, np.min(image_s), np.max(image_s)))
                    print("[image_t] shape={}, dtype={}, min={}, max={}".format(image_t.shape, image_t.dtype, np.min(image_t), np.max(image_t)))

                #====================================================
                # 学習処理
                #====================================================
                output = sess.run(train_op, feed_dict = {image_s_holder: image_s, image_t_holder: image_t} )
                if( args.debug and n_prints > 0 ):
                    print( "output", output )
                    print("[output] shape={}, dtype={}, min={}, max={}".format(output.shape, output.dtype, np.min(output), np.max(output)))

                # 損失関数の計算
                loss_G = sess.run(loss_op, feed_dict = {image_t_holder: output, image_t_holder: image_t} )

                #====================================================
                # 学習過程の表示
                #====================================================
                if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                    #summary = sess.run(board_merge_op, feed_dict = {image_s_holder: image_s, image_t_holder: image_t} )

                    # lr
                    pass

                    # loss
                    print( "epoch={}, step={}, loss_G={:.5f}".format(epoch, step, loss_G) )


                    # visual images
                    pass
                    #tf.summary.image( "train/image_s", image_s, step=step+1 )
                    #tf.summary.image( "train/image_t", image_t, step=step+1 )
                    """
                    visuals = [
                        [ image_s, image_t, output ],
                    ]
                    board_add_images(board_train, 'train', visuals, step+1 )
                    """

                    writer.add_summary(summary)

            except tf.errors.OutOfRangeError:
                print('finished!')
                break

            #====================================================
            # valid データでの処理
            #====================================================
            if( step != 0 and ( step % args.n_display_valid_step == 0 ) ):
                loss_G_total = 0
                n_valid_loop = 0
                """                
                for i, (image_s, image_t) in enumerate(ds_valid):
                    #---------------------------------
                    # 推論処理
                    #---------------------------------
                    pass

                    #---------------------------------
                    # 生成画像表示
                    #---------------------------------
                    if( i <= args.n_display_valid ):
                        with board_train.as_default():
                            visuals = [
                                [ image_s, image_t, output ],
                            ]
                            board_add_images(board_valid, 'valid/{}'.format(i), visuals, step+1 )                            

                    n_valid_loop += 1

                # loss 値表示
                with board_train.as_default():
                    tf.summary.scalar("G/loss_G", loss_G_total/n_valid_loop, step=step+1, description="生成器の全loss")
                """

            step += 1
            iters += args.batch_size
            n_prints -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            pass

    print("Finished Training Loop.")
