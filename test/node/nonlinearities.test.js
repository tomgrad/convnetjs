import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

function tanhLayer() {
  return new convnetjs.TanhLayer({ in_sx: 1, in_sy: 1, in_depth: 1 });
}

test('tanh saturates to +1 and -1 without producing NaN', () => {
  const pos = tanhLayer().forward(new convnetjs.Vol([400]));
  const neg = tanhLayer().forward(new convnetjs.Vol([-400]));
  assert.ok(isFinite(pos.w[0]), 'tanh(400) was ' + pos.w[0]);
  assert.ok(isFinite(neg.w[0]), 'tanh(-400) was ' + neg.w[0]);
  assert.ok(Math.abs(pos.w[0] - 1) < 1e-9);
  assert.ok(Math.abs(neg.w[0] + 1) < 1e-9);
});

test('tanh matches the analytic value for small inputs', () => {
  const out = tanhLayer().forward(new convnetjs.Vol([0.5]));
  assert.ok(Math.abs(out.w[0] - Math.tanh(0.5)) < 1e-9);
});
