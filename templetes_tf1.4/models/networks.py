# -*- coding:utf-8 -*-
import os
import numpy as np

# TensorFlow ライブラリ
import tensorflow as tf

class TempleteNetworks():
    """
    ダミー用のネットワーク
    """
    def __init__( self, out_dim = 1 ):
        super(TempleteNetworks, self).__init__()
        return

    def init_weight_variable( self, input_shape ):
        """
        重みの初期化を行う。
        重みは TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                重みの Variable を初期化するための Tensor の形状
        [Output]
            正規分布に基づく乱数で初期化された重みの Variable 
            session.run(...) はされていない状態。
        """
        # tf.truncated_normal(...) : Tensor を正規分布なランダム値で初期化する
        init_tsr = tf.truncated_normal( shape = input_shape, stddev = 0.01 )

        # 重みの Variable
        weight_var = tf.Variable( init_tsr )
        return weight_var

    def init_bias_variable( self, input_shape ):
        """
        バイアス項 b の初期化を行う。
        バイアス項は TensorFlow の Variable で定義することで、
        学習過程（最適化アルゴリズム Optimizer の session.run(...)）で自動的に TensorFlow により、変更される値となる。
        [Input]
            input_shape : [int,int]
                バイアス項の Variable を初期化するための Tensor の形状
        [Output]
            ゼロ初期化された重みの Variable 
            session.run(...) はされていない状態。
        """
        init_tsr = tf.random_normal( shape = input_shape )

        # バイアス項の Variable
        bias_var = tf.Variable( init_tsr )
        return bias_var

    def __call__( self, image_s_holder ):
        #----------------------------------------------------------------------
        # 計算グラフの構築
        #----------------------------------------------------------------------
        with tf.name_scope( self.__class__.__name__ ):
            # 重みの Variable の list に、１つ目の畳み込み層の重み（カーネル）を追加
            weight = self.init_weight_variable( input_shape = [ 1, 1, 1, 3 ] ) 
            
            # 畳み込み層のオペレーター
            conv_op1 = tf.nn.conv2d(
                        input = image_s_holder,
                        filter = weight,             # 畳込み処理で input で指定した Tensor との積和に使用する filter 行列（カーネル）
                        strides = [ 1, 1, 1, 1 ],    # strides[0] = strides[3] = 1. とする必要がある
                        padding = "SAME"             # ゼロパディングを利用する場合はSAMEを指定
                    )

            output_op = conv_op1
 
        return output_op
