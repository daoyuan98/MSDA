#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

import numpy as np

import os

from flip_gradient import *
from config import *
from model.MDAN import  MDAN

os.environ['KMP_DUPLICATE_LIB_OK']='True'


mnist = load_data(setname='mnist')
svhn  = load_data(setname='svhn')
synth = load_data(setname='synth')
mnistm= load_data(setname='mnistm')


#the last is target domain; others are source domains
datasets = [mnistm, mnist, synth, svhn]

print('tar: ', datasets[-1].name)

combined_train, combined_train_label = get_train(datasets)  # stacked properly.

tar_test = datasets[-1].test.data
tar_test_label = datasets[-1].test.label

# Compute pixel mean for normalizing data
pixel_mean = combined_train.mean((0, 1, 2))

num_test = test_n

sec = np.vstack([np.tile([1., 0.], [num_test // 6, 1]),
                                  np.tile([0., 1.], [num_test // 6, 1])])

combined_test_domain = np.vstack([sec, sec, sec])

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MDAN(pixel_mean)

    learning_rate = tf.placeholder(tf.float32, [])

    domain_loss_ = tf.convert_to_tensor([
        tf.reduce_mean(model.domain_loss_1),
        tf.reduce_mean(model.domain_loss_2),
        tf.reduce_mean(model.domain_loss_3)
    ])

    pred_loss = tf.reduce_mean(model.pred_loss)#scalar
    domain_loss = tf.reduce_mean(domain_loss_)
    total_loss = tf.add(domain_loss_, pred_loss)
    gamma = tf.constant(GAMMA)
    total_loss = tf.multiply(tf.log(tf.reduce_sum(tf.exp(tf.multiply(gamma, total_loss))))/gamma, weight)

    regular_train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(total_loss)
    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

    correct_domain_pred_1 = tf.equal(tf.argmax(model.domain1, 1), tf.argmax(model.pred_1, 1))
    correct_domain_pred_2 = tf.equal(tf.argmax(model.domain2, 1), tf.argmax(model.pred_2, 1))
    correct_domain_pred_3 = tf.equal(tf.argmax(model.domain3, 1), tf.argmax(model.pred_3, 1))

    domain_acc_1 = tf.reduce_mean(tf.cast(correct_domain_pred_1, tf.float32))
    domain_acc_2 = tf.reduce_mean(tf.cast(correct_domain_pred_2, tf.float32))
    domain_acc_3 = tf.reduce_mean(tf.cast(correct_domain_pred_3, tf.float32))

    domain_acc = [domain_acc_1, domain_acc_2, domain_acc_3]


def train_and_evaluate(training_mode, graph, model, num_steps=10000, verbose=False):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        gen_source_batch = batch_generator([combined_train, combined_train_label],   batch_size)

        domain_labels_single_batch_1 = np.vstack([np.tile([1., 0.], [batch_size // 6, 1]),
                                                  np.tile([0., 1.], [batch_size // 6, 1])])

        domain_labels_single_batch_2 = np.vstack([np.tile([1., 0.], [batch_size // 6, 1]),
                                                  np.tile([0., 1.], [batch_size // 6, 1])])

        domain_labels_single_batch_3 = np.vstack([np.tile([1., 0.], [batch_size // 6, 1]),
                                                  np.tile([0., 1.], [batch_size // 6, 1])])

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = head / (1. + 10 * p) ** 0.75

            X, y = next(gen_source_batch)

            _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                [dann_train_op, total_loss,  domain_loss, pred_loss, domain_acc, label_acc],
                feed_dict={
                           model.X: X, model.y: y,
                           model.domain1: domain_labels_single_batch_1,
                           model.domain2: domain_labels_single_batch_2,
                           model.domain3: domain_labels_single_batch_3,
                           model.train: True, model.l: l, learning_rate: lr
                           })

            if i % 5000 == 1 and i > 500:

                print('[test]testing...')
                target_acc = sess.run(label_acc,
                                      feed_dict={model.X: tar_test, model.y: tar_test_label,
                                                 model.train: False})
                print('[test]target_label_acc: ', target_acc)

            if verbose and i % 1000 == 0:
                print('i: {:5d} loss: {:10.6f} dloss:{:8.6f} ploss:{:8.6f}  d_acc: {}  p_acc: {:6f}  p: {:.6f}  l: {:.6f}  lr: {:.6f}'.format(
                    i, batch_loss, dloss, ploss, d_acc, p_acc, p, l, lr))

        target_acc = sess.run(label_acc,
                              feed_dict={model.X: tar_test, model.y: tar_test_label,
                                             model.train: False})
        print('[Final test]target_label_acc: ',target_acc)

        test_emb = sess.run(model.feature, feed_dict={model.X: tar_test})

    return target_acc,  test_emb

def eval(n):

    print('\nnum_steps: ', n, " Model: ", str(model))
    target_acc,  dann_emb = train_and_evaluate(
        'dann', graph, model,
        num_steps=n, verbose=True
    )
    print('Target (MNIST-M) accuracy:', target_acc)
    print('----------------------------')

eval(step)