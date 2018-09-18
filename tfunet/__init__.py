

import os
import logging
import shutil
import numpy as np

from collections import OrderedDict
import tensorflow as tf

from tfunet.layer import (
    Conv2d,
    Deconv2d,
    MaxPool2d,
    PixelWiseSoftMax
)

from tfunet.cost import (
    cross_entropy
)

from tfunet.utils import (
    save_image,
    crop_to_shape,
    combine_img_prediction,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


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


class TFUnet(object):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_layers=3,
                 n_filters=16,
                 filter_size=3,
                 pool_size=2,
                 summary=True,
                 cost="cross_entropy",
                 cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.summary = summary

        self._input_size = 1000

        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, None, None, self.n_channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, self.n_classes], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)

        # build model
        logits, self.params, self.offset = self.build_model(self.x)

        # get cost node
        self.cost = self._get_cost(logits, cost, cost_kwargs)

        # get gradients
        self.gradient_node = self._get_gradient(self.cost, self.params)

        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, self.n_classes]),
                                               tf.reshape(PixelWiseSoftMax()(logits), [-1, self.n_classes]))

        with tf.name_scope("results"):
            self.predicter = PixelWiseSoftMax()(logits)
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_input_node(self, x):
        with tf.name_scope("inputnode"):
            n_cols = tf.shape(x)[1]
            n_rows = tf.shape(x)[2]
            x_image = tf.reshape(x, tf.stack([-1, n_cols, n_rows, self.n_channels]))
            input_node = x_image
            batch_size = tf.shape(x_image)[0]
        return input_node

    def encode(self, input_node):

        self._size = self._input_size
        for l in range(0, self.n_layers):
            with tf.name_scope(f"encode.{l}"):
                n_filters_l = 2 ** l * self.n_filters
                stddev = np.sqrt(2 / (self.filter_size ** 2 * n_filters_l))
                if l == 0:
                    w1 = WeightVariable(name="w.1")(
                        [self.filter_size, self.filter_size, self.n_channels, n_filters_l], stddev)
                else:
                    w1 = WeightVariable(name="w.1")(
                        [self.filter_size, self.filter_size, n_filters_l//2, n_filters_l], stddev)

                w2 = WeightVariable(name="w.2")(
                    [self.filter_size, self.filter_size, n_filters_l, n_filters_l], stddev)

                b1 = BiasVariable(name="b.1")([n_filters_l])
                b2 = BiasVariable(name="b.2")([n_filters_l])

                conv1 = Conv2d(name="conv.1")(input_node, w1, b1, self.keep_prob)
                conv1_ = tf.nn.relu(conv1)
                conv2 = Conv2d(name="conv.2")(conv1_, w2, b2, self.keep_prob)
                self.encoder_outputs[l] = tf.nn.relu(conv2)

                self.weights.append((w1, w2))
                self.biases.append((b1, b2))
                self.conv_outputs.append((conv1, conv2))

                self._size -= 4
                if l < self.n_layers - 1:
                    # self.pool_outputs[l] = max_pool(self.encoder_outputs[l], self.pool_size)
                    self.pool_outputs[l] = MaxPool2d(name="pool")(
                        self.encoder_outputs[l], pool_size=self.pool_size)
                    input_node = self.pool_outputs[l]
                    self._size /= 2

        return self.encoder_outputs[self.n_layers - 1]

    def decode(self, input_node):

        for l in range(self.n_layers - 2, -1, -1):
            with tf.name_scope(f"decode.{l}"):
                n_filters_l = 2 ** (l + 1) * self.n_filters
                stddev = np.sqrt(2 / (self.filter_size ** 2 * n_filters_l))

                wd = WeightVariable(name="w.d")(
                    [self.pool_size, self.pool_size, n_filters_l // 2, n_filters_l], stddev)
                bd = BiasVariable(name="b.d")([n_filters_l // 2])

                deconv_ = tf.nn.relu(Deconv2d(name="deonv")(input_node, wd, stride_size=self.pool_size) + bd)
                deconv_concat_ = crop_and_concat(self.encoder_outputs[l], deconv_)
                self.deconv_outputs[l] = deconv_concat_

                w1 = WeightVariable(name="w.1")(
                    [self.filter_size, self.filter_size, n_filters_l, n_filters_l // 2], stddev)
                w2 = WeightVariable(name="w.2")(
                    [self.filter_size, self.filter_size, n_filters_l // 2, n_filters_l // 2], stddev)

                b1 = BiasVariable(name="b.1")([n_filters_l // 2])
                b2 = BiasVariable(name="b.2")([n_filters_l // 2])

                conv1 = Conv2d(name="conv.1")(deconv_concat_, w1, b1, self.keep_prob)
                conv1_ = tf.nn.relu(conv1)
                conv2 = Conv2d(name="conv.2")(conv1_, w2, b2, self.keep_prob)
                conv2_ = tf.nn.relu(conv2)
                input_node = conv2_
                self.decoder_outputs[l] = input_node

                self.weights.append((w1, w2))
                self.biases.append((b1, b2))
                self.conv_outputs.append((conv1, conv2))

                self._size *= 2
                self._size -= 4

        return self.decoder_outputs[l]

    def classify(self, input_node):
        with tf.name_scope("classify"):
            stddev = np.sqrt(2 / (self.filter_size ** 2 * self.n_classes))
            w = WeightVariable(name="w.out")([1, 1, self.n_filters, self.n_classes], stddev)
            b = BiasVariable(name="b.out")([self.n_classes])
            conv = Conv2d(name="conv.out")(input_node, w, b, tf.constant(1.0))
            output_map = tf.nn.relu(conv)
            self.decoder_outputs["out"] = output_map
        return output_map

    def build_summary(self):
        with tf.name_scope("summary"):
            for i, (c1, c2) in enumerate(self.conv_outputs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in self.pool_outputs.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(self.pool_outputs[k]))

            for k in self.deconv_outputs.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(self.deconv_outputs[k]))

            for k in self.encoder_outputs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', self.encoder_outputs[k])

            for k in self.decoder_outputs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', self.decoder_outputs[k])

    def build_model(self, x):
        """
        Build a new TFUnet model given input tensor variable.
        :param x: input tensor, shape [?, n_cols, n_rows, n_channels]
        """

        logging.info(
            f"Layers {self.n_layers}, " +
            f"features {self.n_filters}, " +
            f"filter size {self.filter_size}x{self.filter_size}, " +
            f"pool size: {self.pool_size}x{self.pool_size}")

        self.weights = []
        self.biases = []
        self.conv_outputs = []
        self.pool_outputs = OrderedDict()
        self.deconv_outputs = OrderedDict()
        self.encoder_outputs = OrderedDict()
        self.decoder_outputs = OrderedDict()

        input_node = self.get_input_node(x)
        encoded = self.encode(input_node)
        decoded = self.decode(encoded)
        output_map = self.classify(decoded)

        if self.summary:
            self.build_summary()

        params = []
        for w1, w2 in self.weights:
            params.append(w1)
            params.append(w2)

        for b1, b2 in self.biases:
            params.append(b1)
            params.append(b2)

        return output_map, params, int(self._input_size - self._size)

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.n_classes])
            flat_labels = tf.reshape(self.y, [-1, self.n_classes])
            if cost_name == "cross_entropy":
                class_weights = cost_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif cost_name == "dice_coefficient":
                eps = 1e-5
                prediction = PixelWiseSoftMax()(logits)
                intersection = tf.reduce_sum(prediction * self.y)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
                loss = -(2 * intersection / (union))

            else:
                raise ValueError("Unknown cost function: " % cost_name)

            regularizer = cost_kwargs.pop("regularizer", None)
            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.params])
                loss += (regularizer * regularizers)

            return loss

    def _get_gradient(self, cost, params):
        return tf.gradients(cost, params)
    
    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param verification_batch_size: size of verification batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    def __init__(self, net, batch_size=1, verification_batch_size = 4, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.verification_batch_size = verification_batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradient_node)]), name="norm_gradients")

        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=False, prediction_path='prediction'):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider(self.verification_batch_size)
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)

                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradient_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: crop_to_shape(batch_y, pred_shape),
                                   self.net.keep_prob: dropout})

                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                    crop_to_shape(batch_y, pred_shape))

                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)

                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape

        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                  self.net.y: crop_to_shape(batch_y, pred_shape),
                                                  self.net.keep_prob: 1.})

        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,
                                                                                   crop_to_shape(batch_y,
                                                                                                      prediction.shape)),
                                                                        loss))

        img = combine_img_prediction(batch_x, batch_y, prediction)
        save_image(img, "%s/%s.jpg" % (self.prediction_path, name))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.predicter],
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                           loss,
                                                                                                           acc,
                                                                                                           error_rate(
                                                                                                               predictions,
                                                                                                               batch_y)))


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
            (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

