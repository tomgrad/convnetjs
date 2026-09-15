'use strict';
import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

function randomVol(sx, sy, depth) {
  const v = new convnetjs.Vol(sx, sy, depth, 0.0);
  for (let i = 0; i < v.w.length; i++) { v.w[i] = Math.random() * 2 - 1; }
  return v;
}

function worstRelativeGradientError(makeNet, makeX, label) {
  const net = makeNet();
  const trainer = new convnetjs.SGDTrainer(net,
    { learning_rate: 0.0001, momentum: 0.0, batch_size: 1, l2_decay: 0.0 });
  const x = makeX();
  trainer.train(x, label);

  const delta = 1e-6;
  let worst = 0;
  for (let i = 0; i < x.w.length; i++) {
    const analytic = x.dw[i];
    const old = x.w[i];
    x.w[i] = old + delta; const c0 = net.getCostLoss(x, label);
    x.w[i] = old - delta; const c1 = net.getCostLoss(x, label);
    x.w[i] = old;
    const numeric = (c0 - c1) / (2 * delta);
    const rel = Math.abs(analytic - numeric) / Math.abs(analytic + numeric + 1e-12);
    if (rel > worst) { worst = rel; }
  }
  return worst;
}

test('gradient check: fully connected', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
      { type: 'fc', num_neurons: 5, activation: 'tanh' },
      { type: 'fc', num_neurons: 5, activation: 'tanh' },
      { type: 'softmax', num_classes: 3 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 2), 1) < 1e-2);
});

test('gradient check: conv + pool + relu', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 4, out_sy: 4, out_depth: 1 },
      { type: 'conv', sx: 3, filters: 2, stride: 1, pad: 1, activation: 'relu' },
      { type: 'pool', sx: 2, stride: 2 },
      { type: 'softmax', num_classes: 2 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(4, 4, 1), 0) < 1e-2);
});

test('gradient check: maxout', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 4 },
      { type: 'fc', num_neurons: 4, activation: 'maxout', group_size: 2 },
      { type: 'softmax', num_classes: 2 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 4), 0) < 1e-2);
});
