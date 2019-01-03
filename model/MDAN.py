from flip_gradient import *
from utils import *
import tensorflow as tf
from config import batch_size

class MDAN(object):

    """
    Multiple Source Domain Adaptation Model for Digits Classification
    """

    def __init__(self, pixel_mean, gamma=10., soft=True):

        self.gamma = gamma
        self.pixel_mean = pixel_mean
        self._build_model(soft)


    def __str__(self):
        return self.__doc__

    def _build_model(self, soft):

        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])

        self.domain1 = tf.placeholder(tf.float32, [None, 2])
        self.domain2 = tf.placeholder(tf.float32, [None, 2])
        self.domain3 = tf.placeholder(tf.float32, [None, 2])

        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        X_input = (tf.cast(self.X, tf.float32) - self.pixel_mean) / 255.

        with tf.variable_scope('feature_extractor'):

            W_conv0 = weight_variable([3, 3, 3, 64])
            b_conv0 = bias_variable([64])
            h_conv0 = tf.add(conv2d(X_input, W_conv0), b_conv0)
            h_conv0 = tf.contrib.layers.batch_norm(h_conv0)
            h_conv0 = tf.nn.relu(h_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([3, 3, 64, 128])
            b_conv1 = bias_variable([128])
            h_conv1 = tf.add(conv2d(h_pool0, W_conv1), b_conv1)
            h_conv1 = tf.contrib.layers.batch_norm(h_conv1)
            h_conv1 = tf.nn.relu(h_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([3, 3, 128, 256])
            b_conv2 = bias_variable([256])
            self.h_conv2 = tf.add(tf.nn.relu(conv2d(h_pool1, W_conv2)), b_conv2)

        with tf.variable_scope('classifier'):

            h_pool3 = max_pool_2x2(self.h_conv2)

            W_conv3 = weight_variable([3, 3, 256, 256])
            b_conv3 = bias_variable([256])
            h_conv3 = tf.add(conv2d(h_pool3, W_conv3), b_conv3)
            h_conv3 = tf.contrib.layers.batch_norm(h_conv3)
            h_pool4 = tf.nn.relu(h_conv3)

            self.feature = tf.reshape(h_pool4, [-1, 4 * 4 * 256])

            all_features = lambda: self.feature
            source_features = lambda: tf.concat([
                tf.slice(self.feature, [0, 0], [batch_size // 6, -1]),
                tf.slice(self.feature, [batch_size // 3, 0], [batch_size // 6, -1]),
                tf.slice(self.feature, [2 * batch_size // 3, 0], [batch_size // 6, -1])
                ], 0)
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.concat([
                tf.slice(self.y, [0, 0], [batch_size // 6, -1]),
                tf.slice(self.y, [batch_size // 3, 0], [batch_size // 6, -1]),
                tf.slice(self.y, [2 * batch_size // 3, 0], [batch_size // 6, -1])
                ], 0)
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([4 * 4 * 256, 2048])
            b_fc0 = bias_variable([2048])
            h_fc0 = tf.add(tf.matmul(classify_feats, W_fc0), b_fc0)
            h_fc0 = tf.contrib.layers.batch_norm(h_fc0)
            h_fc0 = tf.nn.relu(h_fc0)

            W_fc1 = weight_variable([2048, 1024])
            b_fc1 = bias_variable([1024])
            h_fc1 = tf.add(tf.matmul(h_fc0, W_fc1), b_fc1)
            h_fc1 = tf.contrib.layers.batch_norm(h_fc1)
            h_fc1 = tf.nn.relu(h_fc1)

            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            h_fc2 = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)
            h_fc2 = tf.contrib.layers.batch_norm(h_fc2)
            logits = tf.nn.relu(h_fc2)

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.classify_labels)

        with tf.variable_scope('domain_classifier'):

            #Domain 1
            pool_1 = max_pool_2x2(self.h_conv2)

            feature_1_ = flip_gradient(tf.reshape(pool_1, [-1, 4 * 4 * 256]), self.l)
            feature_1 = tf.slice(feature_1_, [0, 0], [batch_size // 3, -1])

            w_fc1_1 = weight_variable([4 * 4 * 256, 2048])
            b_fc1_1 = bias_variable([2048])
            out_1_1 = tf.nn.relu(tf.add(tf.matmul(feature_1, w_fc1_1), b_fc1_1))
            out_1_1 = tf.contrib.layers.batch_norm(out_1_1)

            w_fc2_1 = weight_variable([2048, 2048])
            b_fc2_1 = bias_variable([2048])
            out_2_1 = tf.nn.relu(tf.add(tf.matmul(out_1_1, w_fc2_1), b_fc2_1))
            out_2_1 = tf.contrib.layers.batch_norm(out_2_1)

            w_fc3_1 = weight_variable([2048, 2])
            b_fc3_1 = bias_variable([2])
            logits_1 = tf.add(tf.matmul(out_2_1, w_fc3_1), b_fc3_1)
            logits_1 = tf.contrib.layers.batch_norm(logits_1)

            self.pred_1 = tf.nn.softmax(logits_1)
            self.domain_loss_1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_1, labels=self.domain1)

            #Domain 2
            pool_2 = max_pool_2x2(self.h_conv2)

            feature_2_ = flip_gradient(tf.reshape(pool_2, [-1, 4 * 4 * 256]), self.l)
            feature_2 = tf.slice(feature_2_, [batch_size // 3, 0], [batch_size // 3, -1])

            w_fc1_2 = weight_variable([4 * 4 * 256, 2048])
            b_fc1_2 = bias_variable([2048])
            out_1_2 = tf.nn.relu(tf.add(tf.matmul(feature_2, w_fc1_2), b_fc1_2))
            out_1_2 = tf.contrib.layers.batch_norm(out_1_2)

            w_fc2_2 = weight_variable([2048, 2048])
            b_fc2_2 = bias_variable([2048])
            out_2_2 = tf.nn.relu(tf.add(tf.matmul(out_1_2, w_fc2_2), b_fc2_2))
            out_2_2 = tf.contrib.layers.batch_norm(out_2_2)

            w_fc3_2 = weight_variable([2048, 2])
            b_fc3_2 = bias_variable([2])
            logits_2 = tf.add(tf.matmul(out_2_2, w_fc3_2), b_fc3_2)
            logits_2 = tf.contrib.layers.batch_norm(logits_2)

            self.pred_2 = tf.nn.softmax(logits_2)
            self.domain_loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_2, labels=self.domain2)


            #Domain 3
            pool_3 = max_pool_2x2(self.h_conv2)

            feature_3 = flip_gradient(tf.reshape(pool_3, [-1, 4 * 4 * 256]))
            feature_3 = tf.slice(feature_3, [2 * batch_size //3, 0], [batch_size // 3, -1])

            w_fc1_3 = weight_variable([4 * 4 * 256, 2048])
            b_fc1_3 = bias_variable([2048])
            out_1_3 = tf.nn.relu(tf.add(tf.matmul(feature_3, w_fc1_3), b_fc1_3))
            out_1_3 = tf.contrib.layers.batch_norm(out_1_3)

            w_fc2_3 = weight_variable([2048, 2048])
            b_fc2_3 = bias_variable([2048])
            out_2_3 = tf.nn.relu(tf.add(tf.matmul(out_1_3, w_fc2_3), b_fc2_3))
            out_2_3 = tf.contrib.layers.batch_norm(out_2_3)

            w_fc3_3 = weight_variable([2048, 2])
            b_fc3_3 = bias_variable([2])
            logits_3 = tf.add(tf.matmul(out_2_3, w_fc3_3), b_fc3_3)
            logits_3 = tf.contrib.layers.batch_norm(logits_3)

            self.pred_3 = tf.nn.softmax(logits_3)
            self.domain_loss_3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_3, labels = self.domain3)
