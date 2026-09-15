import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('Vol.toJSON serializes w as a JSON array', () => {
  const v = new convnetjs.Vol([1, 2, 3]);
  const roundTripped = JSON.parse(JSON.stringify(v.toJSON()));

  assert.ok(Array.isArray(roundTripped.w), 'w was ' + JSON.stringify(roundTripped.w));
  assert.deepStrictEqual(roundTripped.w, [1, 2, 3]);
});

test('Vol.fromJSON round-trips values and dimensions', () => {
  const original = new convnetjs.Vol(2, 1, 2, 0.0);
  original.w[0] = 5; original.w[1] = 6; original.w[2] = 7; original.w[3] = 8;

  const restored = new convnetjs.Vol(0, 0, 0, 0.0);
  restored.fromJSON(JSON.parse(JSON.stringify(original.toJSON())));

  assert.strictEqual(restored.sx, 2);
  assert.strictEqual(restored.depth, 2);
  assert.deepStrictEqual(Array.prototype.slice.call(restored.w), [5, 6, 7, 8]);
});
