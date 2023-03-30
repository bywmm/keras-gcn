from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf

from tensorflow.python.ops import sparse_ops


class GraphAggregation(Layer):
    def __init__(self,
                 activation=None,
                 activity_regularizer=None,
                 **kwargs):
        # if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #     kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphAggregation, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        shape_a, _ = input_shapes
        raise NotImplementedError
        return shape_a

    # def compute_output_shape(self, input_shapes):
    #     assert isinstance(input_shapes, list)
    #     shape_a, _ = input_shapes
    #     return shape_a

    def build(self, input_shapes):
        self.build = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1]
        # output = tf.sparse.sparse_dense_matmul(basis, features)

        output = K.dot(tf.sparse.to_dense(basis), features)
        return self.activation(output)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
        }

        base_config = super(GraphAggregation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
