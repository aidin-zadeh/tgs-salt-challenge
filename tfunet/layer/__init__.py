

import tensorflow as tf

from tfunet.utils import (
    to_snake_case
)


class Layer:

    def __init__(self, **kwargs):

        self._valid_kwargs = {
            "name"
        }

        for kwarg in kwargs:
            if kwarg not in self._valid_kwargs:
                raise TypeError(f"Invalid key-word argument: {kwarg}")

        name = kwargs.get("name")
        if not name:
            name = to_snake_case(self.__class__.__name__)

        self.name = name


class Conv2d(Layer):

    def __init__(self, padding="VALID", **kwargs):

        super(Conv2d, self).__init__(**kwargs)

        self.padding = padding

    def __call__(self, x, weight, bias, keep_prob):

        with tf.name_scope("conv2d"):
            conv2d_act = tf.nn.conv2d(x,
                                      weight,
                                      strides=[1, 1, 1, 1],
                                      padding=self.padding)
            conv2d_act = tf.nn.bias_add(conv2d_act, bias)
            return tf.nn.dropout(conv2d_act, keep_prob)


class Deconv2d(Layer):

    def __init__(self, padding="VALID", **kwargs):
        super(Deconv2d, self).__init__(**kwargs)

        self.padding = padding

    def __call__(self, x, weights, stride_size=1):

        with tf.name_scope("deconv2d"):
            in_shape = tf.shape(x)

            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]//2])
            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]//2])
            return tf.nn.conv2d_transpose(x,
                                          weights,
                                          out_shape,
                                          strides=[1, stride_size, stride_size, 1],
                                          padding=self.padding,
                                          name="conv2d.transpose")


class MaxPool2d(Layer):

    def __init__(self, padding="VALID", **kwargs):

        super(MaxPool2d, self).__init__(**kwargs)
        self.padding = padding

    def __call__(self, x, pool_size, **kwargs):

        with tf.name_scope("maxpool2d"):
            return tf.nn.max_pool(x,
                                  ksize=[1, pool_size, pool_size, 1],
                                  strides=[1, pool_size, pool_size, 1],
                                  padding=self.padding)


class PixelWiseSoftMax(Layer):

    def __init__(self, **kwargs):

        super(PixelWiseSoftMax, self).__init__(**kwargs)

    def __call__(self, x):

        with tf.name_scope("pixel.wise.softmax"):
            max_axis = tf.reduce_max(x, axis=3, keepdims=True)
            exp_x = tf.exp(x - max_axis)
            sum_exp_x = tf.reduce_sum(exp_x, axis=3, keepdims=True)
            return exp_x / sum_exp_x










