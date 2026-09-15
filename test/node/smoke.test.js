'use strict';
import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('the built library exposes the public API', () => {
  assert.strictEqual(typeof convnetjs.Net, 'function');
  assert.strictEqual(typeof convnetjs.Vol, 'function');
  assert.strictEqual(typeof convnetjs.Trainer, 'function');

  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ]);
  const out = net.forward(new convnetjs.Vol([0.1, 0.2]));
  assert.strictEqual(out.w.length, 2);
});
