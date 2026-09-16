import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

// Layers after makeLayers: 0=input, 1=fc, 2=fc (added by regression), 3=regression.
// We pin all weights so the first Adam step is analytically known.
function makePinnedNet() {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 1 },
    { type: 'fc', num_neurons: 1 },
    { type: 'regression', num_neurons: 1 }
  ]);
  net.layers[1].filters[0].w[0] = 0.0;
  net.layers[1].biases.w[0] = 0.0;
  net.layers[2].filters[0].w[0] = 1.0;
  net.layers[2].biases.w[0] = 0.0;
  return net;
}

test('adam takes a first step of roughly learning_rate against a unit gradient', () => {
  const net = makePinnedNet();
  const trainer = new convnetjs.Trainer(net, {
    method: 'adam',
    learning_rate: 0.1,
    batch_size: 1,
    l2_decay: 0.0,
    l1_decay: 0.0
  });

  // x=1, y=1 => network output 0 => gradient -1 for both fc weight and bias.
  // First Adam step = -lr * mhat/(sqrt(vhat)+eps) = -0.1 * (-1)/(1+eps) ~ +0.1
  trainer.train(new convnetjs.Vol([1.0]), [1.0]);

  assert.ok(Math.abs(net.layers[1].filters[0].w[0] - 0.1) < 1e-4,
    'weight step was ' + net.layers[1].filters[0].w[0]);
  assert.ok(Math.abs(net.layers[1].biases.w[0] - 0.1) < 1e-4,
    'bias step was ' + net.layers[1].biases.w[0]);
});

test('regression training accepts number and Float64Array targets without warning', () => {
  const original = console.log;
  const logged = [];
  console.log = (...args) => { logged.push(args.join(' ')); };
  try {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 1 },
      { type: 'regression', num_neurons: 1 }
    ]);
    const trainer = new convnetjs.Trainer(net, { learning_rate: 0.01, batch_size: 1 });

    trainer.train(new convnetjs.Vol([1.0]), 1.0); // single-number target
    trainer.train(new convnetjs.Vol([1.0]), new Float64Array([1.0]));
  } finally {
    console.log = original;
  }
  assert.deepStrictEqual(logged, []);
});

test('regression training warns for an unsupported target type', () => {
  const original = console.log;
  const logged = [];
  console.log = (...args) => { logged.push(args.join(' ')); };
  try {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 1 },
      { type: 'regression', num_neurons: 1 }
    ]);
    const trainer = new convnetjs.Trainer(net, { learning_rate: 0.01, batch_size: 1 });
    trainer.train(new convnetjs.Vol([1.0]), 'not-a-target');
  } finally {
    console.log = original;
  }
  assert.ok(logged.some((m) => m.includes('regression')), 'expected a warning, got ' + JSON.stringify(logged));
});
