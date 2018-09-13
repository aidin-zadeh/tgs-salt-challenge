

import tensorflow as tf
import numpy as np
from collections import OrderedDict

from tfunet.layer import (
    Conv2d,
    Deconv2d,
    MaxPool2d,
    PixelWiseSoftMax,
)


EPS = 1e-5


class WeightVariable(object):

    def __init__(self, **kwargs):

        super(WeightVariable, self).__init__()
        name = kwargs.get("name")

        if not name:
            name = "weights"
        self.name = name

    def __call__(self, shape, stddev=0.1):

        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=self.name)


class BiasVariable(object):

    def __init__(self, **kwargs):

        super(BiasVariable, self).__init__()

        name = kwargs.get("name")
        if not name:
            name = "bias"

        self.name =name

    def __call__(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=self.name)


def crop_and_concat(x_1, x_2):
    with tf.name_scope("crop_and_concat"):
        shape_1 = tf.shape(x_1)
        shape_2 = tf.shape(x_2)
        # offsets for the top left corner of the crop
        offsets = [0, (shape_1[1] - shape_2[1]) // 2, (shape_1[2] - shape_2[2]) // 2, 0]
        size = [-1, shape_2[1], shape_2[2], -1]
        inputs_1_cropped = tf.slice(x_1, offsets, size)

    return tf.concat([inputs_1_cropped, x_2], 3)


class TFUnet(object):

    def __init__(self,
                 n_channels,
                 n_classes,
                 n_layers,
                 n_filters,
                 kernel_size,
                 pool_size,
                 stride_size,
                 loss,
                 **kwargs):

        super(TFUnet, self).__init__()

        name = kwargs.get("name")
        if not name:
            name = "tfunet"

        self.name = name

        self.x = tf.placeholder("float", shape=[None, None, None, n_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, n_classes], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride_size = stride_size

        self.weights = []
        self.biases = []
        self.conv_outputs = []
        self.pool_outputs = OrderedDict()
        self.encoder_outputs = OrderedDict()
        self.decoder_outputs = OrderedDict()
        self.skipline_outputs = OrderedDict()
        self.loss = loss
        self.size_in = 100



    def _setup_input_node(self):
        with tf.name_scope("input.node"):
            self._n_cols = tf.shape(self.x)[0]
            self._n_rows = tf.shape(self.x)[1]
            self.input_batch = tf.reshape(self.x, tf.stack([-1, self._n_cols, self._n_rows, self.n_channels]))
            self.batch_size = tf.shape(self.input_batch)[0]

    def _build_encoder(self):

        x_l = self.input_batch  # input to layer `l`
        self.size = self.size_in
        for l in range(0, self.n_layers):
            with tf.name_scope(f"encoder.layer.{l}"):
                n_filters_l = 2 ** l * self.n_filters
                stddev = np.sqrt(2 / (self.kernel_size ** 2 * n_filters_l))
                if l == 0:
                    w_1 = WeightVariable(name="w.1")(
                        [self.kernel_size, self.kernel_size, self.n_channels, n_filters_l], stddev)
                else:
                    w_1 = WeightVariable(name="w.1")(
                        [self.kernel_size, self.kernel_size, n_filters_l//2, n_filters_l], stddev)

                w_2 = WeightVariable(name="w.2")(
                        [self.kernel_size, self.kernel_size, n_filters_l, n_filters_l], stddev)

                b_1 = BiasVariable(name="b.1")([n_filters_l])
                b_2 = BiasVariable(name="b.2")([n_filters_l])

                conv_1 = Conv2d(name="conv.1")(x_l, w_1, b_1, self.keep_prob)
                conv_1_ = tf.nn.relu(conv_1)

                conv_2 = Conv2d(name="conv.2")(conv_1_, w_2, b_2, self.keep_prob)
                conv_2_ = tf.nn.relu(conv_2)

                self.encoder_outputs[l] = conv_2
                self.conv_outputs.append((conv_1, conv_2))
                self.weights.append((w_1, w_2))
                self.biases.append((b_1, b_2))

                self.size -= 4
                if l < self.n_layers-1:
                    self.pool_outputs[l] = MaxPool2d()(self.encoder_outputs[l], self.pool_size)
                    x_l = self.pool_outputs[l]
                    self.size /= 2

        return x_l

    def _build_decoder(self):
        # move backward from the very last layer and apply deconvolution
        x_l = self.encoder_outputs[self.n_layers-1]
        for l in range(self.n_layers-2, -1, -1):
            with tf.name_scope(f"decoder.layer.{l}"):
                n_filters_l = 2 ** (l+1) * self.n_filters
                stddev = np.sqrt(2 / (self.kernel_size ** 2 * n_filters_l))

                w_d = WeightVariable(name=f"w.d")(
                    [self.pool_size, self.pool_size, n_filters_l // 2, n_filters_l], stddev)
                b_d = BiasVariable(name=f"b.d")([n_filters_l // 2])

                w_1 = WeightVariable(name=f"w.1")(
                    [self.kernel_size, self.kernel_size, n_filters_l, n_filters_l//2], stddev)
                w_2 = WeightVariable(name=f"w.2")(
                    [self.kernel_size, self.kernel_size, n_filters_l // 2, n_filters_l // 2], stddev)
                b_1 = BiasVariable(name="b.1")([n_filters_l // 2])
                b_2 = BiasVariable(name="b.2")([n_filters_l // 2])

                # deconvolution - from layer [l-1] to [l]
                deconv = Deconv2d(name="deconv")(x_l, w_d, self.stride_size)
                deconv_ = tf.nn.relu(deconv)
                # concatenation - encoder output from layer [l] + deconvolution output to layer [l]
                deconv_concat_ = crop_and_concat(self.encoder_outputs[l], deconv_)
                # convolutions
                conv_1 = Conv2d(name="conv.1")(deconv_concat_, w_1, b_1, self.keep_prob)
                conv_1_ = tf.nn.relu(conv_1)
                conv_2 = Conv2d(name="conv.2")(conv_1_, w_2, b_2, self.keep_prob)
                conv_2_ = tf.nn.relu(conv_2)

                x_l = conv_2_
                self.decoder_outputs[l] = x_l

                self.conv_outputs.append((conv_1, conv_2))
                self.weights.append((w_1, w_2))
                self.biases.append((b_1, b_2))

                self.size *= 2
                self.size -= 4

        return x_l

    def _build_classifier(self):

        x_out = self.decoder_outputs[0]
        with tf.name_scope(name="classifier"):
            stddev = np.sqrt(2 / (self.kernel_size ** 2 * self.n_classes))
            w = WeightVariable(name="out.w")(
                [1, 1, self.n_filters, self.n_classes], stddev)
            b = BiasVariable(name="out.b")([self.n_classes])
            conv = Conv2d(name="out.conv")(x_out, w, b, tf.constant(1.0))
            conv_ = tf.nn.relu(conv)
            self.decoder_outputs["out"] = conv_
        return conv_

    def _build_model(self):

        # setup input batch
        self._setup_input_node()
        encode = self._build_encoder()
        decode = self._build_decoder()
        logits = self._build_classifier()

        params = []
        for (w_1, w_2), (b_1, b_2) in zip(self.weights, self.biases):
            params.append(w_1)
            params.append(b_1)
            params.append(w_2)
            params.append(b_2)

        return logits, params, int(self.size_in - self.size)

    def get_cost(self, logits, params, loss):

        with tf.name_scope("cost"):
            logits_flat = tf.reshape(logits, [-1, self.n_classes])
            labels_flat = tf.reshape(self.y, [-1 , self.n_classes])

            if loss["name"] == "cross-entropy":
                class_weights = loss.pop("class_weights", None)

                if class_weights:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
                    labels_flat_weighted = tf.multiply(labels_flat, class_weights)
                    labels_flat_weighted = tf.reduce_sum(labels_flat_weighted, axis=1)
                    cost_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_flat, labels=labels_flat)
                    cost_weighted = tf.multiply(cost_map, labels_flat_weighted)
                    cost = tf.reduce_mean(cost_weighted)

                else:
                    cost_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_flat, labels=labels_flat)
                    cost = tf.reduce_mean(cost_map)

            elif loss["name"] == "dice-coefficient":
                labels_predicted = PixelWiseSoftMax()(logits)
                intersections = tf.reduce_sum(labels_predicted * self.y)
                union = EPS + tf.reduce_sum(labels_predicted) + tf.reduce_sum(self.y)
                cost = -(2 * intersections / (union))
            else:
                raise ValueError(f"Invalid loss function: {loss['name']}")

            regularizer = loss.pop("regularizer", None)
            if regularizer:
                regularizer = sum([tf.nn.l2_loss(param) for param in params])
                cost += (regularizer * regularizer)

        return cost

    def get_gradients(self, cost):
        return tf.gradients(cost, self.params)

    def build(self):

        logits, self.params, self.offset = self._build_model()

        self.cost = self.get_cost(logits, self.params, self.loss)
        self.gradients = self.get_gradients(self.cost)

        with tf.name_scope("evaluate"):
            self.y_hat = PixelWiseSoftMax()(logits)
            self.true_positives = tf.equal(tf.arg_maxs(self.y_hat, 3), tf.arg_max(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.true_positives, tf.float32))

    def summary(self):
        pass

    def compile(self):
        pass


loss = {"name": "cross-entropy"}
tfnet = TFUnet(n_channels=3,
               n_classes=4,
               n_layers=3,
               n_filters=10,
               kernel_size=3,
               pool_size=2,
               stride_size=1,
               loss=loss)

tfnet.build()

# WeightVariable(name="aa")