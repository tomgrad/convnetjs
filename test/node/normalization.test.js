import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('lrn applies sane defaults when options are omitted', () => {
  const layer = new convnetjs.LocalResponseNormalizationLayer({
    in_sx: 1, in_sy: 1, in_depth: 3
  });
  const out = layer.forward(new convnetjs.Vol([0.3, -0.7, 0.5]));
  assert.ok(Array.from(out.w).every(Number.isFinite), 'output was ' + Array.from(out.w));
});
