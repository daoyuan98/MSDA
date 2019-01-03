from tensorflow.examples.tutorials.mnist import input_data
import pickle as pkl
import numpy as np
from PIL import Image

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
train_label = mnist.train.labels


mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
test_label  = mnist.test.labels


for i in range(10):
    idx = np.random.randint(0, 100)
    print(train_label[idx])
    img = Image.fromarray(mnist_train[idx])
    img.show()
    input()
    img.close()


with open('data/mnist_data.pkl', 'wb') as f:
    pkl.dump(
        {'train': mnist_train, 'train_label': train_label, 'test_label': test_label, 'test': mnist_test},
        f, pkl.HIGHEST_PROTOCOL
    )