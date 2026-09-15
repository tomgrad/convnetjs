'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('softmax loss stays finite when the true class underflows to zero', () => {
  const layer = new convnetjs.SoftmaxLayer({ in_sx: 1, in_sy: 1, in_depth: 2 });
  layer.forward(new convnetjs.Vol([0, 1000])); // P(class 0) underflows to 0
  const loss = layer.backward(0);
  assert.ok(isFinite(loss), 'loss was ' + loss);
  assert.ok(loss > 0);
});
