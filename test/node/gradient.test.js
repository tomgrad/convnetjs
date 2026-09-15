import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEED = 53;

// Layers initialize their weights with Math.random(), so seed the global RNG
// as well to make net construction (and thus the whole check) reproducible.
Math.random = mulberry32(SEED);

function randomVol(sx, sy, depth, rand) {
  const v = new convnetjs.Vol(sx, sy, depth, 0.0);
  for (let i = 0; i < v.w.length; i++) { v.w[i] = rand() * 2 - 1; }
  return v;
}

function worstRelativeGradientError(makeNet, makeX, label) {
  const net = makeNet();
  const trainer = new convnetjs.SGDTrainer(net,
    { learning_rate: 0.0001, momentum: 0.0, batch_size: 1, l2_decay: 0.0 });
  const x = makeX();
  trainer.train(x, label);
  const analyticGrads = Array.from(x.dw);

  const delta = 1e-6;
  let worst = 0;
  for (let i = 0; i < x.w.length; i++) {
    const analytic = analyticGrads[i];
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
  const rand = mulberry32(SEED);
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 2, rand), 1) < 1e-2);
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
  const rand = mulberry32(SEED);
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(4, 4, 1, rand), 0) < 1e-2);
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
  const rand = mulberry32(SEED);
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 4, rand), 0) < 1e-2);
});
