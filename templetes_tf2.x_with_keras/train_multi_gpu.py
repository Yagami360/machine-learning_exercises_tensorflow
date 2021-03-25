import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import cv2

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.client import device_lib

# Keras
from tensorflow import keras
import tensorflow.keras.backend as K

# 自作モジュール
from data.dataset import load_dataset, TempleteDataGen
from models.networks import TempleteNetworks
from utils.utils import set_random_seed, numerical_sort
from utils.utils import sava_image_tsr
from utils.utils import board_add_image, board_add_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints/", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス（*.hdf5）")
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
    parser.add_argument('--use_datagen', action='store_true', help="データジェネレータを使用するか否か")
    parser.add_argument("--seed", type=int, default=71)
    #parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--gpu_ids', type=str, default="0", help="使用GPU （例 : 0,1,2,3）")
    parser.add_argument('--use_amp', action='store_true')
    #parser.add_argument('--use_tfdbg', choices=['not_use', 'cli', 'gui'], default="not_use", help="tfdbg使用フラグ")
    #parser.add_argument('--detect_inf_or_nan', action='store_true')
    parser.add_argument('--use_tensorboard_debugger', action='store_true', help="TensorBoard Debugger V2 有効化フラグ（）")
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
        print( "device_lib.list_local_devices() : ", device_lib.list_local_devices() )

    if( int(tf.__version__.split(".")[0]) < 2 and int(tf.__version__.split(".")[1]) < 3 and args.use_tensorboard_debugger ):
        print( "TensorBoard Debugger V2 is not supported in tensoflow version {}".format(tf.__version__) )

    # 出力フォルダの作成
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )
    if not os.path.isdir("_debug"):
        os.mkdir("_debug")

    #================================
    # tensorflow 設定
    #================================
    # デバイスの配置ログを有効化
    """
    if( args.debug ):
        tf.debugging.set_log_device_placement(True)
    """

    # TensorBoard Debugger V2 有効化
    if( args.use_tensorboard_debugger ):
        tf.debugging.experimental.enable_dump_debug_info(
            dump_root = os.path.join(args.tensorboard_dir, args.exper_name + "_debug"),
            tensor_debug_mode = "FULL_HEALTH",
            circular_buffer_size = -1
        )

    # AMP 有効化
    if( args.use_amp ):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # Eager execution mode / tensorflow 2.x では明示不要
    #tf.enable_eager_execution()

    # seed 値の固定
    set_random_seed(args.seed)

    # multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()
    #mirrored_strategy = tf.distribute.MirroredStrategy(tf.config.experimental.list_physical_devices("GPU"))

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    if( args.use_datagen ):
        datagen_train = TempleteDataGen( 
            dataset_dir = args.dataset_dir, 
            datamode =  "train",
            image_height = args.image_height, image_width = args.image_width, batch_size = args.batch_size,
        )
    else:
        image_s_trains, image_t_trains, image_s_valids, image_t_valids = load_dataset( args.dataset_dir, image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = args.batch_size, seed = args.seed )
        if( args.debug ):
            print( "[image_s_trains] shape={}, dtype={}, min={}, max={}".format(image_s_trains.shape, image_s_trains.dtype, np.min(image_s_trains), np.max(image_s_trains)) )
            print( "[image_t_trains] shape={}, dtype={}, min={}, max={}".format(image_t_trains.shape, image_t_trains.dtype, np.min(image_t_trains), np.max(image_t_trains)) )
            print( "[image_s_valids] shape={}, dtype={}, min={}, max={}".format(image_s_valids.shape, image_s_valids.dtype, np.min(image_s_valids), np.max(image_s_valids)) )
            print( "[image_t_valids] shape={}, dtype={}, min={}, max={}".format(image_t_valids.shape, image_t_valids.dtype, np.min(image_t_valids), np.max(image_t_valids)) )


    #================================
    # モデルの構造を定義する。
    #================================
    with mirrored_strategy.scope():
        model_G = TempleteNetworks(out_dim=3)
        model_G( tf.zeros([args.batch_size, args.image_height, args.image_width, 3], dtype=tf.float32) )


    #================================
    # loss 設定
    #================================
    with mirrored_strategy.scope():
        loss_mse = tf.keras.losses.MeanSquaredError()

    #================================
    # optimizer 設定
    #================================
    with mirrored_strategy.scope():
        optimizer_G = tf.keras.optimizers.Adam( learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2 )


    #================================
    # AMP 有効化
    #================================
    if( args.use_amp ):
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    #================================
    # モデルをコンパイル
    #================================
    with mirrored_strategy.scope():
        model_G.compile(
            loss = loss_mse,
            optimizer = optimizer_G,
            metrics = ['mae']
        )

    """
    with mirrored_strategy.scope():
        #================================
        # モデルの構造を定義する。
        #================================
        model_G = TempleteNetworks(out_dim=3)
        model_G( tf.zeros([args.batch_size, args.image_height, args.image_width, 3], dtype=tf.float32) )

        #================================
        # loss 設定
        #================================
        loss_mse = tf.keras.losses.MeanSquaredError()

        #================================
        # optimizer 設定
        #================================
        optimizer_G = tf.keras.optimizers.Adam( learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2 )

        #================================
        # AMP 有効化
        #================================
        if( args.use_amp ):
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)

        #================================
        # モデルをコンパイル
        #================================
        model_G.compile(
            loss = loss_mse,
            optimizer = optimizer_G,
            metrics = ['mae']
        )
    """

    if( args.debug ):
        model_G.summary()

    #================================
    # モデルの読み込み
    #================================
    if( args.load_checkpoints_path ):
        # モデルを定義したあと読み込む場合は load_weights() を使用
        # モデルを定義せずに読み込む場合は keras.models.load_model() を使用
        #model_G = keras.models.load_model(args.load_checkpoints_path)
        model_G.load_weights(args.load_checkpoints_path)
        print( "load checkpoints in `{}`.".format(args.load_checkpoints_path) )
        init_epoch = 0
    else:
        init_epoch = 0

    #================================
    # call backs の設定
    #================================
    # 各エポック終了毎のモデルのチェックポイント保存用 call back
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint( 
        filepath = os.path.join(args.save_checkpoints_dir, args.exper_name, "step_{epoch:05d}.hdf5"), 
        monitor = 'loss', 
        verbose = 2, 
        save_weights_only = True,       # 
        save_best_only = False,         # 精度がよくなった時だけ保存するかどうか指定。False の場合は毎 epoch 毎保存．
        mode = 'auto',                  # 
        period = args.n_save_epoches    # 何エポックごとに保存するか
    )

    # tensorboard の出力用の call back
    callback_board_train = tf.keras.callbacks.TensorBoard( log_dir = os.path.join(args.tensorboard_dir, args.exper_name), write_graph = False )

    callbacks = [ callback_board_train, callback_checkpoint ]
    
    #================================
    # tfdbg でのデバッグ処理有効化
    #================================
    """
    if( args.use_tfdbg == "cli" ):
        # [ToDo] AttributeError: module 'tensorflow.keras.backend' has no attribute 'get_session' のエラー解消
        from tensorflow.python import debug as tf_debug
        import tensorflow.keras.backend as K
        sess = K.get_session()
        #sess.run( tf.global_variables_initializer() )
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if( args.detect_inf_or_nan ):
            from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
            sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    elif( args.use_tfdbg == "gui" ):
        # [ToDo] AttributeError: module 'tensorflow.python.debug' has no attribute 'TensorBoardDebugWrapperSession' のエラー解消
        from tensorflow.python import debug as tf_debug
        from tensorflow.python import debug as tf_debug
        import tensorflow.keras.backend as K
        sess = K.get_session()
        tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6007")
        if( args.detect_inf_or_nan ):
            from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
            sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    """

    #================================
    # モデルの学習
    #================================    
    if( args.use_datagen ):
        history = model_G.fit_generator( 
            generator = datagen_train, 
            epochs = args.n_epoches, 
            steps_per_epoch = len(datagen_train),
            verbose = 1,
            workers = args.n_workers,
            shuffle = True,
            use_multiprocessing = True,
            callbacks = callbacks
        )
    else:
        history = model_G.fit( 
            x = image_s_trains, y = image_t_trains, 
            epochs = args.n_epoches, 
            steps_per_epoch = 1,
            verbose = 1,
            workers = args.n_workers,
            shuffle = True,
            use_multiprocessing = True,
            callbacks = callbacks
        )
