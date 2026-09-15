import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('makeLayers does not mutate the caller layer definitions', () => {
  const defs = [
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ];
  const before = JSON.stringify(defs);

  const net = new convnetjs.Net();
  net.makeLayers(defs);

  assert.strictEqual(JSON.stringify(defs), before);
});

test('fromJSON throws a clear error for an unknown layer type', () => {
  const net = new convnetjs.Net();
  assert.throws(
    () => net.fromJSON({ layers: [{ layer_type: 'bogus' }] }),
    /Unknown layer type: bogus/
  );
});
