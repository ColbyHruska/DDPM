import tensorflow as tf
import math
import keras
from keras import layers
from keras import activations
from keras_unet.models import custom_unet as unet
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

def define_noise_predictor(input_shape, learning_rate, n_layers):
    first_conv_channels = 64

    in_image = layers.Input(input_shape)
    in_time = layers.Input((1))

    size = input_shape[0] * input_shape[1]
    temb = TimeEmbedding(size)(in_time)
    temb = TimeMLP(size)(temb)
    temb = layers.Reshape((input_shape[0], input_shape[1], 1))(temb)

    hid = layers.Concatenate()([in_image, temb])
    hid = unet((input_shape[0], input_shape[1], input_shape[2] + 1), input_shape[2], num_layers=n_layers, activation=activations.swish, output_activation=activations.linear, use_batch_norm=False, dropout=0)(hid)

    model = keras.Model([in_image, in_time], hid)
    opt = Adam(learning_rate=learning_rate)
    model.compile(opt, MeanSquaredError())

    return model
