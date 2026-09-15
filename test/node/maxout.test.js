'use strict';
import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('maxout fromJSON restores a switches buffer sized to the output volume', () => {
  // 2x2 spatial, depth 4, group_size 2 => out_depth 2 => 2*2*2 = 8 switches
  const layer = new convnetjs.MaxoutLayer({ in_sx: 2, in_sy: 2, in_depth: 4, group_size: 2 });
  const json = layer.toJSON();

  const restored = new convnetjs.MaxoutLayer({});
  restored.fromJSON(json);

  assert.strictEqual(restored.switches.length, 8);
});
