# ConvNetJS Guide

ConvNetJS is a small neural-network library for browsers and Node. Networks are described with an
array of **layer definitions** and built with `net.makeLayers(layer_defs)`. All data flows through
`Vol`s — 3D volumes of numbers with a width (`sx`), height (`sy`) and depth (`depth`). Non-image
inputs keep `sx = sy = 1`.

```js
// Node (CommonJS): require('convnetjs'), or require('./build/convnet.cjs') from a local checkout.
// In the browser, load build/convnet.js and use the global `convnetjs` instead.
const convnetjs = require('convnetjs');

const layer_defs = [
  { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
  { type: 'fc', num_neurons: 20, activation: 'relu' },
  { type: 'softmax', num_classes: 10 }
];

const net = new convnetjs.Net();
net.makeLayers(layer_defs);

const x = new convnetjs.Vol([0.3, -0.5]);
const probs = net.forward(x); // Vol with 10 probabilities

const trainer = new convnetjs.SGDTrainer(net, { learning_rate: 0.01, l2_decay: 0.001 });
trainer.train(x, 3); // train toward class 3
```

## Available layers

These are the `type` values recognized by `Net.makeLayers`:

| `type` | Class | Purpose |
|---|---|---|
| `input` | `InputLayer` | Declares the input volume size |
| `fc` | `FullyConnLayer` | Fully connected (dense) layer |
| `conv` | `ConvLayer` | Convolutional layer |
| `pool` | `PoolLayer` | Max-pooling layer |
| `relu` | `ReluLayer` | ReLU activation |
| `sigmoid` | `SigmoidLayer` | Sigmoid activation |
| `tanh` | `TanhLayer` | Tanh activation |
| `maxout` | `MaxoutLayer` | Maxout activation |
| `dropout` | `DropoutLayer` | Dropout regularization |
| `lrn` | `LocalResponseNormalizationLayer` | Local response normalization |
| `softmax` | `SoftmaxLayer` | Softmax classification loss |
| `svm` | `SVMLayer` | Multiclass SVM (hinge) loss |
| `regression` | `RegressionLayer` | L2 regression loss |

### `input`

Declares the size of the input volume. Must be the first layer.

| Option | Default | Notes |
|---|---|---|
| `out_depth` | `0` | Required. Aliases: `depth` |
| `out_sx` | `1` | Aliases: `sx`, `width` |
| `out_sy` | `1` | Aliases: `sy`, `height` |

### `fc` (fully connected)

| Option | Default | Notes |
|---|---|---|
| `num_neurons` | — | Required. Alias: `filters` |
| `activation` | none | One of `relu`, `sigmoid`, `tanh`, `maxout` (adds an activation layer after this one) |
| `group_size` | `2` | Group size when `activation: 'maxout'` |
| `drop_prob` | none | If set, adds a dropout layer after this one |
| `l1_decay_mul` | `0.0` | Multiplier for the trainer's `l1_decay` |
| `l2_decay_mul` | `1.0` | Multiplier for the trainer's `l2_decay` |
| `bias_pref` | `0.0` | Initial bias; `0.1` for `relu` |

### `conv`

| Option | Default | Notes |
|---|---|---|
| `sx` | — | Required. Filter width |
| `sy` | `sx` | Filter height |
| `filters` | — | Required. Number of filters (output depth) |
| `stride` | `1` | Filter stride |
| `pad` | `0` | Zero padding around the input |
| `activation` | none | As for `fc` |
| `group_size` | `2` | As for `fc` |
| `drop_prob` | none | As for `fc` |
| `l1_decay_mul` | `0.0` | As for `fc` |
| `l2_decay_mul` | `1.0` | As for `fc` |
| `bias_pref` | `0.0` | As for `fc` |

Output size is `floor((in + 2*pad - filter) / stride + 1)` per spatial dimension.

### `pool`

| Option | Default | Notes |
|---|---|---|
| `sx` | — | Required. Window width |
| `sy` | `sx` | Window height |
| `stride` | `2` | Window stride |
| `pad` | `0` | Zero padding |

### `relu`, `sigmoid`, `tanh`

Elementwise activations with no options. They are usually added implicitly through the `activation`
option on `fc`/`conv` rather than declared directly.

### `maxout`

| Option | Default | Notes |
|---|---|---|
| `group_size` | `2` | Number of inputs combined by each output; output depth is `floor(in_depth / group_size)` |

### `dropout`

| Option | Default | Notes |
|---|---|---|
| `drop_prob` | `0.5` | Probability of dropping a unit during training |

Usually added implicitly via `drop_prob` on `fc`/`conv`.

### `lrn` (local response normalization)

| Option | Default | Notes |
|---|---|---|
| `k` | — | Required |
| `n` | — | Required. Window size; should be odd |
| `alpha` | — | Required |
| `beta` | — | Required |

### `softmax` / `svm`

| Option | Default | Notes |
|---|---|---|
| `num_classes` | — | Required. Number of classes |

Both automatically insert a fully connected layer of size `num_classes` before the loss.

### `regression`

| Option | Default | Notes |
|---|---|---|
| `num_neurons` | — | Required. Number of regression outputs |

Automatically inserts a fully connected layer of that size before the loss. Training targets may be
an array of values, a single number, or `{ dim, val }`.

## How `layer_defs` are expanded

`makeLayers` "desugars" the definitions before building the layers, so `net.layers` is usually
longer than the array you passed:

- `softmax`/`svm` insert an `fc` layer of size `num_classes`.
- `regression` inserts an `fc` layer of size `num_neurons`.
- `activation` on an `fc`/`conv` layer inserts the corresponding activation layer after it.
- `drop_prob` on a non-dropout layer inserts a `dropout` layer after it.

For example, `[input, fc(relu), softmax]` becomes
`[input, fc, relu, fc, softmax]` — 5 layers.

## Trainers

Trainers update a network's weights. `SGDTrainer` is an alias of `Trainer`; the optimizer is chosen
with the `method` option: `sgd` (default), `adam`, `adagrad`, `adadelta`, `windowgrad`, or
`nesterov`. Common options: `learning_rate`, `l1_decay`, `l2_decay`, `batch_size`, `momentum`.
