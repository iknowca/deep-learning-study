import sys, os
sys.path.append(os.pardir+'/MNIST/dataset')
print(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print("x_train: ", x_train.shape)
print("t_train: ", t_train.shape)


train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print("x_batch: ", x_batch)
print("t_batch: ", t_batch)

def cross_entropy_error(y, t, delta=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(np.arrange(batch_size)+delta))/batch_size