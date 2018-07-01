# This is a network layers file for convenient use
# yibo

import tensorflow as tf
import ops
import numpy as np
import pruning_ops as pops


def add_cov_layer(input_x, layer_numble, filter_size, activation_function = None):

  layer_name = "convolutional_layer%s" % layer_numble

  with tf.name_scope(layer_name):

    with tf.name_scope("Filter"):
      Filter = tf.Variable(tf.random_normal(
        filter_size, stddev = 0.1), trainable = False, dtype = tf.float32, name = "filter")
      tf.summary.histogram(layer_name + "/filter", Filter)

    with tf.name_scope("Bias"):
      bias = tf.Variable(tf.constant(0.1, shape = [filter_size[3]]),
        trainable = False, dtype = tf.float32, name = "bias")
      tf.summary.histogram(layer_name + "/bias", bias)

    with tf.name_scope("convolution"):
      conv = tf.nn.conv2d(input_x, Filter, strides=[1, 1, 1, 1], padding = "VALID")

    with tf.name_scope('conv_add_b'):
      temp = tf.add(conv, bias)
    if activation_function is None:
      output_y = temp
    else:
      output_y = activation_function(temp)
    tf.summary.histogram(layer_name + "/output", output_y)
    return output_y
