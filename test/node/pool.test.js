import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('pool backprops finitely even when a window is entirely padding', () => {
  // With pad=2 and a 1x1 input, some 2x2 windows cover only padding and record
  // the -1 sentinel switch. Backprop must not try to write to a negative index.
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 1 },
    { type: 'pool', sx: 2, sy: 2, stride: 1, pad: 2 },
    { type: 'regression', num_neurons: 1 }
  ]);
  const pool = net.layers[1];
  const x = new convnetjs.Vol(1, 1, 1, 0.0);
  x.w[0] = 1.0;

  net.forward(x, true);
  assert.ok(Array.from(pool.switchx).some((v) => v === -1),
    'expected at least one fully-padded window');

  net.backward([0.0]);
  assert.ok(Array.from(x.dw).every(Number.isFinite));
});
