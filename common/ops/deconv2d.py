"""

"""

import numpy as np
import tensorflow as tf
from common.ops.sn import spectral_normed_weight


def Deconv2D(inputs, in_channels, output_channels, filter_size, stride=2, padding='SAME', he_init=True,
             spectral_normed=False, update_collection=None, inputs_norm=False, biases=True, name='Conv2D'):
    """

    Args:

    Returns:
      tensor of shape (batch_size, 2*height, 2*width, output_channels)
    """
    with tf.variable_scope(name):
        fan_in = in_channels * filter_size ** 2 / (stride ** 2)
        # fan_out = output_channels * filter_size ** 2
        if inputs_norm:
            inv_c = np.sqrt(2.0 / fan_in)
            inputs_ = inputs * inv_c
        else:
            inputs_ = inputs

        if he_init:
            # initializer = tf.initializers.he_uniform()
            initializer = tf.initializers.he_normal()
            # initializer = tf.variance_scaling_initializer(
            #     scale=2., mode="fan_in", distribution="truncated_normal", seed=None)
        else:  # Normalized init (Glorot & Bengio)
            # initializer = tf.glorot_uniform_initializer()
            initializer = tf.glorot_normal_initializer()
            # initializer = tf.variance_scaling_initializer(
            #     scale=1., mode="fan_avg", distribution="truncated_normal", seed=None)

        filters = tf.get_variable(
            name='Filters', shape=[filter_size, filter_size, output_channels, in_channels], dtype=tf.float32,
            initializer=initializer)

        if spectral_normed:
            with tf.variable_scope('filters'):
                filters = spectral_normed_weight(filters, update_collection=update_collection)

        input_shape = tf.shape(inputs_)
        output_shape = tf.stack([input_shape[0], 2 * input_shape[1], 2 * input_shape[2], output_channels])

        result = tf.nn.conv2d_transpose(
            value=inputs_,
            filter=filters,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format="NHWC"
        )

        if biases:
            _biases = tf.get_variable(
                name='Biases', shape=[output_channels, ], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            result = tf.nn.bias_add(result, _biases)

        return result
