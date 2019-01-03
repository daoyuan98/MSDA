import pickle as pkl
import scipy.io as sio
import numpy as np
from PIL import Image

def truncate(pics, labels):
    n = pics.shape[3]
    res = [0]*n
    labels_ = np.zeros((n,10),np.uint8)
    for i in range(n):
        print(np.array(pics[:,:,:,i]).shape)
        pic = Image.fromarray(pics[:,:,:,i])
        pic = pic.resize((28,28))
        res[i] = np.asarray(pic)
        labels_[i][labels[i]%10] = 1
    return np.array(res), labels_

train = sio.loadmat('data/SVHN/train_32x32.mat')
test = sio.loadmat('data/SVHN/test_32x32.mat')

train_data = np.array(train['X'])
train_label = np.array(train['y'][:,0])

test_data = np.array(test['X'])
test_label = np.array(test['y'][:,0])

train_data, train_label_ = truncate(train_data, train_label)
test_data, test_label_  = truncate(test_data, test_label)

print(train_data[0].shape)
print(test_data[0].shape)

for i in range(10):
    idx = np.random.randint(0, 100)
    print(train_label[idx], train_label_[idx])
    print(test_label[idx], test_label_[idx])
    img = Image.fromarray(train_data[idx])
    img.show()
    print(train_label[idx])
    input()
    img.close()



with open('./data/svhn_data.pkl', 'wb') as f:
    pkl.dump({ 'train': train_data, 'train_label':train_label_, 'test': test_data, 'test_label':test_label_}, f, pkl.HIGHEST_PROTOCOL)


