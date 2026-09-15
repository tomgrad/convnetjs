import os
import pickle

import numpy
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, 'mnist.pkl'), 'rb') as f:
    # The MNIST pickle was written by Python 2, so decode its byte strings as latin1.
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# 20 training batches (50000 train + 10000 validation) plus 1 test batch of 3000.
# Each row of the resulting image is one flattened 28x28 sample.
x = numpy.concatenate((train_set[0], valid_set[0], test_set[0][:3000]))
x = numpy.round(x * 255).astype(numpy.uint8)

for i in range(20):
    batch = x[3000 * i:3000 * (i + 1), :]
    Image.fromarray(batch).save(os.path.join(HERE, 'mnist_batch_%d.png' % i))
Image.fromarray(x[60000:, :]).save(os.path.join(HERE, 'mnist_batch_20.png'))

# dump the labels (70000 of them; the demo only uses the first 63000)
labels = numpy.concatenate((train_set[1], valid_set[1], test_set[1])).tolist()
with open(os.path.join(HERE, 'mnist_labels.js'), 'w') as f:
    f.write('var labels=' + str(labels) + ';\n')
