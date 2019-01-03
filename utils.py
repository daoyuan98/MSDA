from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.axes_grid1 import ImageGrid
from Dataset import Dataset
from PIL import Image
from config import N_TRAIN, N_TEST, batch_size

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, constraint=tf.keras.constraints.max_norm(4))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='SAME'
    )


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        padding='SAME'
    )


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def get_indices(indices, i, l_domain, seg):

    index = np.random.choice(l_domain, seg)
    if i % 2 == 0: # extract src indices:
        index += (i//2)*l_domain
    else: #extract from tar
        index += l_domain * 3
    # print(index)
    indices.extend(index)


def batch_generator(data, batch_size, shuffle=False):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    adomain = data[0].shape[0]//4
    apart = batch_size // 6
    #
    # |----|----|----|----|
    while True:
        indices = []
        for i in range(6):
            get_indices(indices, i, adomain, apart)
        yield data[0][indices], data[1][indices]

        #start = np.random.randint(0, data.shape[0] - batch_size)
        #end = start + batch_size

        #yield data[start:end]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def load_data(setname='mnist'):
    #path = '/Users/didi/PycharmProjects/t/data/' + setname + '_data.pkl'
    path = '/data/' + setname + '_data.pkl'
    data = pkl.load(open(path, 'rb'))

    data_train = np.array(data['train'][:N_TRAIN]).astype(np.uint8)
    data_test  = np.array(data['test'][:N_TEST]).astype(np.uint8)

    data_train_label = np.array(data['train_label'][:N_TRAIN])
    data_test_label  =  np.array(data['test_label'])[:N_TEST]

    return Dataset(data_train, data_test, data_train_label, data_test_label, setname)


def get_train(datasets):

    data = []
    label = []

    for dataset in datasets:
        data.append(dataset.train.data)
        label.append(dataset.train.label)

    return np.array(data).reshape([-1, 28, 28, 3]), np.array(label).reshape([-1, 10])

def get_test(datasets):
    n = len(datasets[0]) // batch_size
    a_batch = batch_size // 6

    data = []
    label = []

    print(n)

    tar = datasets[-1]

    for i in range(n):

        begin = i * a_batch
        end = begin + a_batch

        data.append(tar.test.data[begin:end])
        label.append(tar.test.label[begin:end])

    return np.array(data).reshape([-1, 28, 28, 3]), np.array(label).reshape([-1, 10])


def check_datasets(images, labels, n=10):
    for i in range(n):
        idx = np.random.randint(0, len(images))
        img = Image.fromarray(images[idx])
        label = labels[idx]
        img.show()
        print(label)
        input()
        img.close()

