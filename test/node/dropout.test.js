import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('dropout prediction scales activations by the keep probability', () => {
  // Training zeroes each unit with probability drop_prob, so a unit's expected
  // activation is (1 - drop_prob) * x. Prediction must reproduce that scaling.
  const layer = new convnetjs.DropoutLayer({
    in_sx: 1, in_sy: 1, in_depth: 1, drop_prob: 0.2
  });
  const out = layer.forward(new convnetjs.Vol([1.0]), false);
  assert.ok(Math.abs(out.w[0] - 0.8) < 1e-12, 'prediction output was ' + out.w[0]);
});

test('dropout prediction is identity when nothing is dropped', () => {
  const layer = new convnetjs.DropoutLayer({
    in_sx: 1, in_sy: 1, in_depth: 1, drop_prob: 0.0
  });
  const out = layer.forward(new convnetjs.Vol([2.5]), false);
  assert.strictEqual(out.w[0], 2.5);
});
