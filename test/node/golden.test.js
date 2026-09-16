import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

// Layers: 0=input, 1=fc, 2=relu, 3=fc (added by softmax), 4=softmax.
// All weights are pinned so every value below is analytically known.
function makePinnedNet() {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ]);
  net.layers[1].filters[0].w.set([1, 0]); net.layers[1].biases.w[0] = 0.5;
  net.layers[1].filters[1].w.set([0, 1]); net.layers[1].biases.w[1] = -0.25;
  net.layers[1].filters[2].w.set([1, 1]); net.layers[1].biases.w[2] = 0.0;
  net.layers[3].filters[0].w.set([1, 0, 1]); net.layers[3].biases.w[0] = 0.0;
  net.layers[3].filters[1].w.set([0, 1, 1]); net.layers[3].biases.w[1] = 0.0;
  return net;
}

test('golden: forward, backward and gradients are unchanged', () => {
  const net = makePinnedNet();
  const x = new convnetjs.Vol([0.5, -0.25]);

  const out = net.forward(x);
  assert.ok(Math.abs(out.w[0] - 0.7310585786300049) < 1e-12);
  assert.ok(Math.abs(out.w[1] - 0.2689414213699951) < 1e-12);

  const loss = net.backward(1);
  assert.ok(Math.abs(loss - 1.3132616875182228) < 1e-12);

  // gradient at the input
  assert.ok(Math.abs(x.dw[0] - 0.7310585786300049) < 1e-12);
  assert.ok(Math.abs(x.dw[1] - 0) < 1e-12);

  // first fc layer parameter gradients
  assert.deepStrictEqual(Array.from(net.layers[1].filters[0].dw),
    [0.36552928931500245, -0.18276464465750122]);
  assert.ok(Math.abs(net.layers[1].biases.dw[0] - 0.7310585786300049) < 1e-12);
  assert.deepStrictEqual(Array.from(net.layers[1].filters[1].dw), [0, 0]);
  assert.deepStrictEqual(Array.from(net.layers[1].filters[2].dw), [0, 0]);

  // second fc layer parameter gradients
  assert.deepStrictEqual(Array.from(net.layers[3].filters[0].dw),
    [0.7310585786300049, 0, 0.18276464465750122]);
  assert.deepStrictEqual(Array.from(net.layers[3].filters[1].dw),
    [-0.7310585786300049, 0, -0.18276464465750122]);
});

test('golden: serialization shape and round-trip are unchanged', () => {
  const net = makePinnedNet();
  const json = JSON.parse(JSON.stringify(net.toJSON()));

  assert.strictEqual(json.layers.length, 5);
  assert.strictEqual(json.layers.map((l) => l.layer_type).join(','),
    'input,fc,relu,fc,softmax');
  assert.ok(Array.isArray(json.layers[1].filters[0].w));
  assert.deepStrictEqual(json.layers[1].filters[0].w, [1, 0]);

  const restored = new convnetjs.Net();
  restored.fromJSON(json);
  const out = restored.forward(new convnetjs.Vol([0.5, -0.25]));
  assert.ok(Math.abs(out.w[0] - 0.7310585786300049) < 1e-12);
});

test('golden: conv/pool serialization round-trip preserves forward output', () => {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 4, out_sy: 4, out_depth: 1 },
    { type: 'conv', sx: 3, filters: 2, stride: 1, pad: 1, activation: 'relu' },
    { type: 'pool', sx: 2, stride: 2 },
    { type: 'softmax', num_classes: 2 }
  ]);
  const x = new convnetjs.Vol(4, 4, 1, 0.0);
  for (let i = 0; i < x.w.length; i++) { x.w[i] = Math.sin(i); }

  const before = Array.from(net.forward(x).w);
  const restored = new convnetjs.Net();
  restored.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  const after = Array.from(restored.forward(x).w);

  assert.deepStrictEqual(after, before);
});

test('golden: the global namespace exposes the public API', () => {
  const names = ['Net', 'Vol', 'Trainer', 'SGDTrainer', 'MagicNet', 'augment',
    'img_to_vol', 'ConvLayer', 'FullyConnLayer', 'PoolLayer', 'InputLayer',
    'RegressionLayer', 'SoftmaxLayer', 'SVMLayer', 'TanhLayer', 'MaxoutLayer',
    'ReluLayer', 'SigmoidLayer', 'DropoutLayer',
    'LocalResponseNormalizationLayer', 'randf', 'randi', 'randn', 'zeros',
    'maxmin', 'randperm', 'weightedSample', 'arrUnique', 'arrContains',
    'getopt', 'assert'];
  for (const name of names) {
    assert.ok(convnetjs[name] !== undefined, 'missing public export: ' + name);
  }
});
