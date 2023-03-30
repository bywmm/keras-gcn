from __future__ import print_function

from keras.engine import Layer
import tensorflow as tf


class BN(Layer):
    def __init__(self,
                 **kwargs):
        super(BN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shapes):
        self.build = True

    def call(self, x, mask=None):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True) + 0.0001)
        return x

    def get_config(self):
        config = {}

        base_config = super(BN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
