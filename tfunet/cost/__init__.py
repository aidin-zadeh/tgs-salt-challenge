

import tensorflow as tf


def cross_entropy(y, y_hat):
    return -tf.reduce_mean(
        y*tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)),
        name="cross_entropy")