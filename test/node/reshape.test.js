import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('reshape flattens a spatial volume preserving flat order', () => {
  const layer = new convnetjs.ReshapeLayer({ in_sx: 2, in_sy: 2, in_depth: 2, sx: 1, sy: 1, depth: 8 });
  const x = new convnetjs.Vol(2, 2, 2, 0.0);
  for (let i = 0; i < x.w.length; i++) { x.w[i] = i; }

  const out = layer.forward(x);
  assert.strictEqual(out.sx, 1);
  assert.strictEqual(out.sy, 1);
  assert.strictEqual(out.depth, 8);
  assert.deepStrictEqual(Array.from(out.w), [0, 1, 2, 3, 4, 5, 6, 7]);
});

test('reshape unflattens a vector into a spatial volume', () => {
  const layer = new convnetjs.ReshapeLayer({ in_sx: 1, in_sy: 1, in_depth: 8, sx: 2, sy: 2, depth: 2 });
  const x = new convnetjs.Vol(1, 1, 8, 0.0);
  for (let i = 0; i < x.w.length; i++) { x.w[i] = i; }

  const out = layer.forward(x);
  assert.strictEqual(out.sx, 2);
  assert.strictEqual(out.sy, 2);
  assert.strictEqual(out.depth, 2);
  assert.deepStrictEqual(Array.from(out.w), [0, 1, 2, 3, 4, 5, 6, 7]);
  // value at (x=1,y=0,d=0) is flat index ((2*0)+1)*2+0 = 2
  assert.strictEqual(out.get(1, 0, 0), 2);
});

test('reshape backward copies gradients through unchanged', () => {
  const layer = new convnetjs.ReshapeLayer({ in_sx: 2, in_sy: 1, in_depth: 2, sx: 1, sy: 1, depth: 4 });
  const x = new convnetjs.Vol(2, 1, 2, 0.0);

  const out = layer.forward(x);
  out.dw.set([1, 2, 3, 4]);
  layer.backward();

  assert.deepStrictEqual(Array.from(x.dw), [1, 2, 3, 4]);
});

test('reshape serializes and round-trips standalone', () => {
  const layer = new convnetjs.ReshapeLayer({ in_sx: 2, in_sy: 2, in_depth: 2, sx: 1, sy: 1, depth: 8 });
  const json = JSON.parse(JSON.stringify(layer.toJSON()));
  assert.strictEqual(json.layer_type, 'reshape');

  const restored = new convnetjs.ReshapeLayer();
  restored.fromJSON(json);
  assert.strictEqual(restored.out_sx, 1);
  assert.strictEqual(restored.out_sy, 1);
  assert.strictEqual(restored.out_depth, 8);

  const x = new convnetjs.Vol(2, 2, 2, 0.0);
  for (let i = 0; i < x.w.length; i++) { x.w[i] = i; }
  assert.deepStrictEqual(Array.from(restored.forward(x).w), Array.from(layer.forward(x).w));
});

test('a net can reshape an fc output into a conv input and round-trips', () => {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 4 },
    { type: 'fc', num_neurons: 4 },
    { type: 'reshape', sx: 2, sy: 2, depth: 1 },
    { type: 'conv', sx: 3, filters: 2, stride: 1, pad: 1, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ]);
  const x = new convnetjs.Vol([0.1, 0.2, 0.3, 0.4]);

  const before = Array.from(net.forward(x).w);
  const restored = new convnetjs.Net();
  restored.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  assert.deepStrictEqual(Array.from(restored.forward(x).w), before);
});

test('reshape rejects a shape whose element count does not match', () => {
  assert.throws(
    () => new convnetjs.ReshapeLayer({ in_sx: 2, in_sy: 2, in_depth: 2, sx: 3, sy: 3, depth: 1 }),
    /cannot reshape/
  );
});

test('reshape rejects missing or non-positive integer dimensions', () => {
  const base = { in_sx: 2, in_sy: 2, in_depth: 2 };
  assert.throws(() => new convnetjs.ReshapeLayer(base), /positive integer/);
  assert.throws(() => new convnetjs.ReshapeLayer({ ...base, sx: 1.5, sy: 1, depth: 8 }), /positive integer/);
  assert.throws(() => new convnetjs.ReshapeLayer({ ...base, sx: 0, sy: 1, depth: 8 }), /positive integer/);
});
