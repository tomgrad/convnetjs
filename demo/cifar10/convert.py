import pickle

import numpy
from PIL import Image


def unpickle(file):
  with open(file, 'rb') as fo:
    # CIFAR-10 ships Python 2 pickles; latin1 keeps the string keys as str.
    return pickle.load(fo, encoding='latin1')


xs = []
ys = []
for j in range(5):
  d = unpickle('data_batch_' + str(j + 1))
  xs.append(d['data'])
  ys.append(d['labels'])

d = unpickle('test_batch')
xs.append(d['data'])
ys.append(d['labels'])

x = numpy.concatenate(xs)
y = numpy.concatenate(ys)

# (N, 3072) -> (N, 1024, 3): the three 1024-long channel blocks become RGB.
# Each sample is a row, so every batch image is 1024 px wide and 1000 px tall.
x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))

for i in range(50):
  Image.fromarray(x[1000 * i:1000 * (i + 1), :]).save(f'cifar10_batch_{i}.png')
Image.fromarray(x[50000:51000, :]).save('cifar10_batch_50.png')  # test set

# dump the labels (tolist() so numpy ints serialize as plain numbers)
L = 'var labels=' + str(y[:51000].tolist()) + ';\n'
with open('cifar10_labels.js', 'w') as f:
  f.write(L)
