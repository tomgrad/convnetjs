import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('predict returns -1 when no candidates have been evaluated', () => {
  const magic = new convnetjs.MagicNet([], []);
  assert.strictEqual(magic.predict(new convnetjs.Vol([1.0])), -1);
});
