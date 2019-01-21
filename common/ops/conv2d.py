"""
Convolution for data in format of 'NWHC'.
"""

import numpy as np
import tensorflow as tf
# import common as lib
from common.ops.sn import spectral_normed_weight


def Conv2D(inputs, input_dim, output_dim, filter_size=3, stride=1, name='Conv2D',
           conv_type='conv2d', channel_multiplier=0, dilation_rate=2, padding='SAME',
           spectral_normed=False, update_collection=None, inputs_norm=False, he_init=True, biases=True):
    """
    Args:
      inputs: Tensor of shape (batch size, height, width, in_channels).
      input_dim: in_channels.
      output_dim:
      filter_size:
      stride: Integer (for [1, stride, stride, 1]) or tuple/list.
      name:
      conv_type: conv2d, depthwise_conv2d, separable_conv2d.
      channel_multiplier:
      padding:
      spectral_normed:
      update_collection:
      inputs_norm: From PGGAN.
      he_init:
      biases:

    Returns:
      tensor of shape (batch_size, out_height, out_width, output_dim)
    """
    # with tf.name_scope(name) as scope:
    with tf.variable_scope(name):
        if conv_type != "conv2d":
            if conv_type == "atrous_conv2d":
                assert (dilation_rate > 0, 'dilation_rate should >0!')
            else:
                assert (channel_multiplier > 0, 'channel_multiplier should >0!')

        fan_in = input_dim * filter_size ** 2
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
            name='Filters', shape=[filter_size, filter_size, input_dim, output_dim], dtype=tf.float32,
            initializer=initializer) * 0.1

        if channel_multiplier > 0:
            depthwise_filters = tf.get_variable(
                name='depthwise_filters', shape=[filter_size, filter_size, input_dim, channel_multiplier],
                dtype=tf.float32, initializer=initializer) * 0.1
            pointwise_filters = tf.get_variable(
                name='pointwise_filters', shape=[1, 1, input_dim * channel_multiplier, output_dim],
                dtype=tf.float32, initializer=initializer) * 0.1

        if spectral_normed:
            with tf.variable_scope('filters'):
                filters = spectral_normed_weight(filters, update_collection=update_collection)

            if channel_multiplier > 0:
                with tf.variable_scope('depthwise_filters'):
                    depthwise_filters = spectral_normed_weight(depthwise_filters, update_collection=update_collection)

                with tf.variable_scope('pointwise_filters'):
                    pointwise_filters = spectral_normed_weight(pointwise_filters, update_collection=update_collection)

        if conv_type == 'conv2d':
            result = tf.nn.conv2d(
                input=inputs_,
                filter=filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                data_format='NHWC'
            )
        elif conv_type == 'depthwise_conv2d':
            result = tf.nn.depthwise_conv2d(
                input=inputs_,
                filter=depthwise_filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                rate=None,
                name=None,
                data_format='NHWC'
            )
        elif conv_type == 'separable_conv2d':
            result = tf.nn.separable_conv2d(
                inputs_,
                depthwise_filter=depthwise_filters,
                pointwise_filter=pointwise_filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                rate=None,
                name=None,
                data_format='NHWC'
            )
        elif conv_type == 'atrous_conv2d':
            result = tf.nn.conv2d(
                input=inputs_,
                filter=filters,
                padding=padding,
                strides=[1, stride, stride, 1],
                dilations=[1, dilation_rate, dilation_rate, 1],
                data_format='NHWC'
            )
        else:
            raise NotImplementedError('{0} is not supported!'.format(conv_type))

        if biases:
            _biases = tf.get_variable(
                name='Biases', shape=[output_dim, ], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            result = tf.nn.bias_add(result, _biases, data_format='NHWC')

        return result
