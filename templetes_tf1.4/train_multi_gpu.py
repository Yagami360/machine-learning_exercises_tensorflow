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
from tensorflow.python.client import device_lib

# 自作モジュール
from data.dataset import Dataset
from models.networks import TempleteNetworks, TempleteNetworksMultiGPU
from utils.utils import set_random_seed, numerical_sort
from utils.utils import sava_image_tsr
#from utils.utils import board_add_image, board_add_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints/", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みファイルのディレクトリ")
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
    parser.add_argument("--n_save_max", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--use_tfrecord', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--gpu_ids', type=str, default="0", help="使用GPU （例 : 0,1,2,3）")
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_tfdbg', choices=['not_use', 'cli', 'gui'], default="not_use", help="tfdbg使用フラグ")
    parser.add_argument('--detect_inf_or_nan', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        str_gpu_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_gpu_id in str_gpu_ids:
            id = int(str_gpu_id)
            if id >= 0:
                args.gpu_ids.append(id)

        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

        print( "tensoflow version : ", tf.__version__ )
        from tensorboard import version
        print( "tensorboard version : ", version.VERSION )
        print( "device_lib.list_local_devices() : ", device_lib.list_local_devices() )

    # 出力フォルダの作成
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )

    # AMP 有効化
    if( args.use_amp ):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # 計算グラフ初期化
    tf.reset_default_graph()

    # Session 開始
    sess = tf.Session()

    # seed 値の固定
    set_random_seed(args.seed)

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train = Dataset( args.dataset_dir, datamode = "train", image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = args.batch_size, shuffle = True, use_tfrecord = args.use_tfrecord, data_augument = args.data_augument )
    ds_valid = Dataset( args.dataset_dir, datamode = "valid", image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = 1, shuffle = False, use_tfrecord = args.use_tfrecord, data_augument = False )

    #================================
    # 変数とプレースホルダを設定
    #================================
    image_s_holder = tf.placeholder(tf.float32, [None, args.image_height, args.image_width, 3], name = "image_s_holder" )
    image_t_holder = tf.placeholder(tf.float32, [None, args.image_height, args.image_width, 3], name = "image_t_holder" )

    #================================
    # モデルの構造を定義する。
    #================================
    model_G = TempleteNetworksMultiGPU(out_dim=3, gpu_ids=args.gpu_ids)
    output_op_gpus = model_G(image_s_holder)

    #================================
    # loss 関数の設定
    #================================
    loss_op_gpus = []
    for gpu_id in args.gpu_ids:
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('loss' + '_gpu_{}'.format(gpu_id)):
                loss_op = tf.reduce_mean( tf.abs(image_t_holder - output_op_gpus[gpu_id]) )
                loss_op_gpus.append(loss_op)

    #================================
    # optimizer の設定
    #================================
    optimizer_op_gpus = []
    train_op_gpus = []
    for gpu_id in args.gpu_ids:
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('optimizer' + '_gpu_{}'.format(gpu_id)):
                optimizer_op = tf.train.AdamOptimizer( learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2 )
                """
                if( args.use_amp ):
                    optimizer_op = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_op)
                """
                train_op = optimizer_op.minimize(loss_op)
                optimizer_op_gpus.append(optimizer_op)
                train_op_gpus.append(train_op)

    #================================
    # 学習済みモデルの読み込み
    #================================
    # モデルの保存用 Saver
    saver = tf.train.Saver(max_to_keep=args.n_save_max)
    if( os.path.exists(args.load_checkpoints_dir) ):
        ckpt_state = tf.train.get_checkpoint_state(args.load_checkpoints_dir)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        print( "load checkpoints in `{}`.".format(ckpt_state.model_checkpoint_path) )
        init_epoch = int(ckpt_state.model_checkpoint_path.split("-")[-1])
        step = init_epoch               # @
        iters = step * args.batch_size  # @
    else:
        init_epoch = 0
        step = 0
        iters = 0

    #================================
    # tensorboard 出力
    #================================
    board_train = tf.summary.FileWriter( os.path.join(args.tensorboard_dir, args.exper_name), sess.graph )
    board_valid = tf.summary.FileWriter( os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )

    board_loss_op = tf.summary.scalar("G/loss_G", loss_op_gpus[0])
    board_train_image_s_op = tf.summary.image( 'train/image_s', image_s_holder, max_outputs = args.batch_size )
    board_train_image_t_op = tf.summary.image( 'train/image_t', image_t_holder, max_outputs = args.batch_size )
    board_train_output_op = tf.summary.image( 'train/output', image_t_holder, max_outputs = args.batch_size )
    board_valid_image_s_op = tf.summary.image( 'valid/image_s', image_s_holder )
    board_valid_image_t_op = tf.summary.image( 'valid/image_t', image_t_holder )
    board_valid_output_op = tf.summary.image( 'valid/output', image_t_holder )
    board_merge_op = tf.summary.merge_all()

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_prints = 1

    # 変数初期化
    sess.run( tf.global_variables_initializer() )

    # tfdbg でのデバッグ処理有効化
    if( args.use_tfdbg == "cli" ):
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if( args.detect_inf_or_nan ):
            from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
            sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    elif( args.use_tfdbg == "gui" ):
        # [ToDo] AttributeError: module 'tensorflow.python.debug' has no attribute 'TensorBoardDebugWrapperSession' のエラー解消
        from tensorflow.python import debug as tf_debug
        tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6007")
        if( args.detect_inf_or_nan ):
            from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
            sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)

    for epoch in tqdm( range(args.n_epoches+init_epoch), desc = "epoches", initial=init_epoch ):
        # データセット初期化
        sess.run(ds_train.init_iter_op)
        while True:
            try:
                # ミニバッチデータの取り出し
                image_s, image_t = sess.run(ds_train.batch_op)

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                if image_s.shape[0] != args.batch_size:
                    break

                if( args.debug and n_prints > 0 ):
                    print("[image_s] shape={}, dtype={}, min={}, max={}".format(image_s.shape, image_s.dtype, np.min(image_s), np.max(image_s)))
                    print("[image_t] shape={}, dtype={}, min={}, max={}".format(image_t.shape, image_t.dtype, np.min(image_t), np.max(image_t)))

                #====================================================
                # 学習処理
                #====================================================
                _, output, loss_G = sess.run([train_op_gpus[0], output_op_gpus[0], loss_op_gpus[0]], feed_dict = {image_s_holder: image_s, image_t_holder: image_t} )
                if( args.debug and n_prints > 0 ):
                    print("[output] shape={}, dtype={}, min={}, max={}".format(output.shape, output.dtype, np.min(output), np.max(output)))

                #====================================================
                # 学習過程の表示
                #====================================================
                if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                    # lr
                    pass

                    # loss
                    print( "epoch={}, step={}, loss_G={:.5f}".format(epoch, step, loss_G) )
                    board_train.add_summary(sess.run(board_loss_op, feed_dict = {image_s_holder: image_s, image_t_holder: image_t} ), global_step=step)

                    # 画像表示
                    board_train.add_summary(sess.run(board_train_image_s_op, feed_dict = {image_s_holder: image_s} ), global_step=step)
                    board_train.add_summary(sess.run(board_train_image_t_op, feed_dict = {image_t_holder: image_t} ), global_step=step)
                    board_train.add_summary(sess.run(board_train_output_op, feed_dict = {image_t_holder: output} ), global_step=step)

                    """
                    visuals = [
                        [ image_s, image_t, output ],
                    ]
                    board_add_images(board_train, 'train', visuals, step+1 )
                    """

            except tf.errors.OutOfRangeError:
                break

            #====================================================
            # valid データでの処理
            #====================================================
            if( step != 0 and ( step % args.n_display_valid_step == 0 ) ):
                sess.run(ds_valid.init_iter_op)

                loss_G_total = 0
                n_valid_loop = 0
                for i in range(100000):
                    try:
                        # ミニバッチデータの取り出し
                        image_s, image_t = sess.run(ds_valid.batch_op)

                        # 出力画像と loss 値取得
                        output, loss_G = sess.run([output_op_gpus[0], loss_op_gpus[0]], feed_dict = {image_s_holder: image_s, image_t_holder: image_t} )
                        loss_G_total += loss_G

                        # 画像表示
                        if( i <= args.n_display_valid ):
                            #board_valid_image_s_op = tf.summary.image( 'valid/image_s/{}'.format(i), image_s_holder, max_outputs=1 )
                            #board_valid_image_t_op = tf.summary.image( 'valid/image_t/{}'.format(i), image_t_holder, max_outputs=1 )
                            #board_valid_output_op = tf.summary.image( 'valid/output/{}'.format(i), image_t_holder, max_outputs=1 )

                            board_valid.add_summary(sess.run(board_valid_image_s_op, feed_dict = {image_s_holder: image_s} ), global_step=step)
                            board_valid.add_summary(sess.run(board_valid_image_t_op, feed_dict = {image_t_holder: image_t} ), global_step=step)
                            board_valid.add_summary(sess.run(board_valid_output_op, feed_dict = {image_t_holder: output} ), global_step=step)
        
                        n_valid_loop += 1

                    except tf.errors.OutOfRangeError:
                        break

                # loss 値出力 : [ToDo] loss total の出力
                board_valid.add_summary(sess.run(board_loss_op, feed_dict = {image_s_holder: image_s, image_t_holder: image_t} ), global_step=step)
                #board_loss_G_total_op = tf.summary.scalar("G/loss_G", loss_G_total/n_valid_loop)
                #board_valid.add_summary(sess.run(board_loss_G_total_op), global_step=step)

            step += 1
            iters += args.batch_size
            n_prints -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            # *.data-00000-of-00001 : 変数名をテンソル値としてマッピングした独自のフォーマット
            # *.index : 複数のstepでデータを保存した際に、同名の「.data-00000-of-00001」ファイルが、どのstepのデータであるのかを一意に定める。
            # *.meta : モデルのネットワーク構造と重みを格納
            saver.save(sess, os.path.join(args.save_checkpoints_dir, args.exper_name, "model_G"), global_step=epoch )

    print("Finished Training Loop.")

    board_train.close()
    board_valid.close()
    sess.close()
