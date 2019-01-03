import pickle as pkl
import scipy.io as sio
from PIL import Image
import numpy as np

def show_from_mat(mat):
    img = Image.fromarray(mat)
    img.show()

def random_bound():
    a1 = b1 = 0
    a2 = b2 = 32
    if np.random.randint(0,2):
        a1 += 4
    else:
        a2 -= 4
    if np.random.randint(0,2):
        b1 += 4
    else:
        b2 -= 4
    return a1,a2, b1, b2

def process(train_path, test_path):
    f = sio.loadmat(train_path)
    train_data_ = np.array(f['X'])
    train_label_ = f['y'][:,0]
    n_pic = train_data_.shape[3]

    train_label = np.zeros((n_pic, 10))


    train_data = [0]*n_pic

    for i in range(20000):
        train_data[i] = np.asarray(Image.fromarray(train_data_[:,:,:,i]).resize((28,28)))
        train_label[i][train_label_[i]] = 1

    f = sio.loadmat(test_path)
    test_data_ = np.array(f['X'])
    test_label_ = f['y'][:,0]
    n_pic = test_data_.shape[3]
    test_label = np.zeros((n_pic, 10))

    test_data = [0]*n_pic

    a1, a2, b1, b2 = random_bound()

    for i in range(9000):
        test_data[i] = test_data_[a1:a2, b1:b2, :, i]
        test_label[i][test_label_[i]] = 1

    assert(len(test_data)==len(test_label) and len(train_data)==len(train_label))

    #test
    for i in range(10):
        n = np.random.randint(100)
        print(test_label[n], test_label_[n])

    if 'small' in train_path:
        file_name = 'data/synth_data_small.pkl'
    else:
        file_name = 'data/synth_data.pkl'

    with open(file_name, 'wb') as f:
        pkl.dump({'train': train_data[:20000], 'train_label': train_label[:20000], 'test': test_data[:9000], 'test_label': test_label[:9000]}, f,
                 pkl.HIGHEST_PROTOCOL)

    print('execute OK!')


process('data/SynthDigits/synth_train_32x32.mat','data/SynthDigits/synth_test_32x32.mat')

def test(n=5):
    data = pkl.load(open('data/synth_data_small.pkl','rb'))
    train_data = data['train']
    train_label = data['train_label']
    for i in range(n):
        show_from_mat(train_data[i])
        print(train_label[i])
        input()

test(10)

