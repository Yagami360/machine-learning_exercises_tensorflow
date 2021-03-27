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
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--use_tfrecord', action='store_true')
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
    # Eager execution mode / tensorflow 2.x では明示不要
    #tf.enable_eager_execution()                    
    
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

    # seed 値の固定
    set_random_seed(args.seed)

    # tensorboard 出力
    board_train = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_valid = tf.summary.create_file_writer( logdir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )
    board_train.set_as_default()
    #board_valid.set_as_default()

    # multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()
    #mirrored_strategy = tf.distribute.MirroredStrategy(tf.config.experimental.list_physical_devices("GPU"))

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train, ds_valid, n_trains, n_valids = load_dataset( args.dataset_dir, image_height = args.image_height, image_width = args.image_width, n_channels = 3, batch_size = args.batch_size, use_tfrecord = args.use_tfrecord, seed = args.seed )
    with mirrored_strategy.scope():
        ds_train = mirrored_strategy.experimental_distribute_dataset(ds_train)
        ds_valid = mirrored_strategy.experimental_distribute_dataset(ds_valid)

    if( args.debug ):
        print( "n_trains : ", n_trains )
        print( "n_valids : ", n_valids )
        print( "ds_train : ", ds_train )
        print( "ds_valid : ", ds_valid )

    #================================
    # モデルの構造を定義する。
    #================================
    with mirrored_strategy.scope():
        model_G = TempleteNetworks(out_dim=3)
        model_G( tf.zeros([args.batch_size, args.image_height, args.image_width, 3], dtype=tf.float32) )
        if( args.debug ):
            model_G.summary()

    #================================
    # モデルの読み込み
    #================================
    if( args.load_checkpoints_dir ):
        ckpt_state = tf.train.get_checkpoint_state(args.load_checkpoints_dir)
        model_G.load_weights(ckpt_state.model_checkpoint_path)
        print( "load checkpoints in `{}`.".format(ckpt_state.model_checkpoint_path) )
        init_epoch = int(ckpt_state.model_checkpoint_path.split("-")[-1])
        step = init_epoch               # @
        iters = step * args.batch_size  # @
    else:
        init_epoch = 0
        step = 0
        iters = 0

    #================================
    # optimizer の設定
    #================================
    with mirrored_strategy.scope():
        optimizer_G = tf.keras.optimizers.Adam( learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2 )

    #================================
    # AMP 有効化
    #================================
    if( args.use_amp ):
        with mirrored_strategy.scope():
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)
            optimizer_G = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer_G, loss_scale='dynamic')

    #================================
    # loss 関数の設定
    #================================
    with mirrored_strategy.scope():
        #loss_fn = tf.keras.losses.MeanSquaredError()
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        def calc_loss_multi_gpus(target, output, batch_size):
            losses_multi_gpus = loss_fn(target, output)
            return tf.nn.compute_average_loss(losses_multi_gpus, global_batch_size=batch_size)

    #================================
    # tfdbg でのデバッグ処理有効化
    #================================
    """
    if( args.use_tfdbg == "cli" ):
        from tensorflow.python import debug as tf_debug
        #import tensorflow.python.keras.backend as K
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
        #import tensorflow.python.keras.backend as K
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
    print("Starting Training Loop...")
    n_prints = 1
    for epoch in tqdm( range(args.n_epoches+init_epoch), desc = "epoches", initial=init_epoch ):
        # ミニバッチデータの取り出し
        for image_s, image_t in ds_train:
            image_s_gpus = image_s.values
            image_t_gpus = image_t.values
            if( args.debug and n_prints > 0 ):
                for i in range(len(args.gpu_ids)):
                    print("[image_s_gpus[{}]] shape={}, dtype={}, min={}, max={}".format(i, image_s_gpus[i].shape, image_s_gpus[i].dtype, np.min(image_s_gpus[i].numpy()), np.max(image_s_gpus[i].numpy())))
                    print("[image_t_gpus[{}]] shape={}, dtype={}, min={}, max={}".format(i, image_t_gpus[i].shape, image_t_gpus[i].dtype, np.min(image_t_gpus[i].numpy()), np.max(image_t_gpus[i].numpy())))
                    sava_image_tsr( image_s_gpus[i][0], "_debug/image_s_gpu{}.png".format(i) )
                    sava_image_tsr( image_t_gpus[i][0], "_debug/image_t_gpu{}.png".format(i) )

            #====================================================
            # 学習処理
            #====================================================
            #@tf.function    # 高速化のためのデコレーター
            def train_on_batch(input, target, batch_size):
                # スコープ以下を自動微分計算
                with tf.GradientTape() as tape:
                    # モデルの forward 処理
                    output = model_G(input, training=True)

                    # 損失関数の計算
                    #loss_G = loss_fn(target, output)
                    loss_G = calc_loss_multi_gpus(target, output, batch_size)
                    #loss_G = tf.reduce_sum(loss_G, keepdims=True) * (1. / 128.0) # 要注意

                # モデルの更新処理
                grads = tape.gradient(loss_G, model_G.trainable_weights)
                optimizer_G.apply_gradients(zip(grads, model_G.trainable_weights))
                #train_acc_metric.update_state(target, logits)
                return output, loss_G

            @tf.function    # 高速化のためのデコレーター
            def train_on_batch_multi_gpus(input, target, batch_size):
                loss_gpus = mirrored_strategy.run(train_on_batch, args=(input, target, batch_size))
                return mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_gpus, axis=None)     # lossのreduceをなくした分ここでreduceする

            output_gpus, loss_G_gpus = train_on_batch_multi_gpus( image_s, image_t, args.batch_size )
            output_gpus, loss_G_gpus = output_gpus.values, loss_G_gpus.values

            loss_G_gpus_total = 0
            for i in range(len(args.gpu_ids)):
                loss_G_gpus_total += loss_G_gpus[i]

            loss_G = loss_G_gpus_total / len(args.gpu_ids)
            output = tf.concat(output_gpus, axis=0)
            if( args.debug and n_prints > 0 ):
                for i in range(len(args.gpu_ids)):
                    print("[output_gpus] shape={}, dtype={}, min={}, max={}".format(output_gpus[i].shape, output_gpus[i].dtype, np.min(output_gpus[i]), np.max(output_gpus[i])))                    
                print("[output] shape={}, dtype={}, min={}, max={}".format(output.shape, output.dtype, np.min(output), np.max(output)))

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # lr
                pass

                # loss
                print( "epoch={}, step={}, loss_G={:.5f}, loss_G_gpus={}".format(epoch, step, loss_G, loss_G_gpus) )
                with board_train.as_default():
                    tf.summary.scalar("G/loss_G", loss_G, step=step+1, description="生成器の全loss")
                    for i in range(len(args.gpu_ids)):
                        tf.summary.scalar("G/loss_G_gpu{}",format(i), loss_G_gpus[i], step=step+1)

                # visual images
                with board_train.as_default():
                    for i in range(len(args.gpu_ids)):
                        visuals = [
                            [ image_s_gpus[i], image_t_gpus[i], output_gpus[i] ],
                        ]
                        board_add_images(board_train, 'train_gpu{}'.format(i), visuals, step+1 )

            #====================================================
            # valid データでの処理
            #====================================================
            """
            if( step != 0 and ( step % args.n_display_valid_step == 0 ) ):
                loss_G_total = 0
                n_valid_loop = 0                
                for i, (image_s, image_t) in enumerate(ds_valid):
                    #---------------------------------
                    # 推論処理
                    #---------------------------------
                    @tf.function
                    def eval_on_batch(input, target):
                        # スコープ以下を自動微分計算
                        with tf.GradientTape() as tape:
                            # モデルの forward 処理
                            output = model_G(input, training=False)

                            # 損失関数の計算
                            loss_G = loss_fn(target, output)

                        return output, loss_G

                    output, loss_G = eval_on_batch( image_s, image_t )
                    loss_G_total += loss_G

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
            model_G.save_weights( os.path.join(args.save_checkpoints_dir, args.exper_name, "model_G-{}".format(epoch)) )

    print("Finished Training Loop.")
