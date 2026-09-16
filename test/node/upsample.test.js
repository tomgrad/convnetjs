import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('upsample nearest repeats each input pixel into a scale x scale block', () => {
  const layer = new convnetjs.UpsampleLayer({ in_sx: 2, in_sy: 2, in_depth: 1, scale: 2 });
  const x = new convnetjs.Vol(2, 2, 1, 0.0);
  x.w.set([1, 2, 3, 4]); // (x,y): (0,0)=1 (1,0)=2 (0,1)=3 (1,1)=4

  const out = layer.forward(x);
  assert.strictEqual(out.sx, 4);
  assert.strictEqual(out.sy, 4);
  assert.strictEqual(out.depth, 1);

  const expected = [
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [3, 3, 4, 4]
  ];
  for (let y = 0; y < 4; y++) {
    for (let xx = 0; xx < 4; xx++) {
      assert.strictEqual(out.get(xx, y, 0), expected[y][xx], 'at (' + xx + ',' + y + ')');
    }
  }
});

test('upsample preserves depth and supports scale 3', () => {
  const layer = new convnetjs.UpsampleLayer({ in_sx: 1, in_sy: 1, in_depth: 2, scale: 3 });
  const x = new convnetjs.Vol(1, 1, 2, 0.0);
  x.w[0] = 0.5;
  x.w[1] = -0.5;

  const out = layer.forward(x);
  assert.strictEqual(out.sx, 3);
  assert.strictEqual(out.sy, 3);
  assert.strictEqual(out.depth, 2);
  for (let p = 0; p < 9; p++) {
    assert.strictEqual(out.w[p * 2], 0.5);
    assert.strictEqual(out.w[p * 2 + 1], -0.5);
  }
});

test('upsample backward accumulates gradients into the source pixel', () => {
  const layer = new convnetjs.UpsampleLayer({ in_sx: 2, in_sy: 1, in_depth: 1, scale: 2 });
  const x = new convnetjs.Vol(2, 1, 1, 0.0);
  x.w[0] = 1;
  x.w[1] = 2;

  const out = layer.forward(x);
  out.dw.set([1, 2, 3, 4]);
  layer.backward();

  // output x=0,1 map to input 0; output x=2,3 map to input 1
  assert.deepStrictEqual(Array.from(x.dw), [3, 7]);
});

test('upsample serializes and round-trips standalone', () => {
  const layer = new convnetjs.UpsampleLayer({ in_sx: 2, in_sy: 3, in_depth: 2, scale: 2 });
  const json = JSON.parse(JSON.stringify(layer.toJSON()));
  assert.strictEqual(json.layer_type, 'upsample');
  assert.strictEqual(json.scale, 2);

  const restored = new convnetjs.UpsampleLayer();
  restored.fromJSON(json);
  assert.strictEqual(restored.out_sx, 4);
  assert.strictEqual(restored.out_sy, 6);
  assert.strictEqual(restored.out_depth, 2);

  const x = new convnetjs.Vol(2, 3, 2, 0.0);
  for (let i = 0; i < x.w.length; i++) { x.w[i] = i; }
  assert.deepStrictEqual(Array.from(restored.forward(x).w), Array.from(layer.forward(x).w));
});

test('a net with an upsample layer round-trips through JSON', () => {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 2, out_sy: 2, out_depth: 1 },
    { type: 'upsample', scale: 2 },
    { type: 'softmax', num_classes: 2 }
  ]);
  const x = new convnetjs.Vol(2, 2, 1, 0.0);
  x.w.set([0.1, 0.2, 0.3, 0.4]);

  const before = Array.from(net.forward(x).w);
  const restored = new convnetjs.Net();
  restored.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  assert.deepStrictEqual(Array.from(restored.forward(x).w), before);
});

test('upsample rejects a missing or invalid scale', () => {
  const base = { in_sx: 2, in_sy: 2, in_depth: 1 };
  assert.throws(() => new convnetjs.UpsampleLayer(base), /integer scale/);
  assert.throws(() => new convnetjs.UpsampleLayer({ ...base, scale: 1.5 }), /integer scale/);
  assert.throws(() => new convnetjs.UpsampleLayer({ ...base, scale: 0 }), /integer scale/);
  assert.throws(() => new convnetjs.UpsampleLayer({ ...base, scale: -2 }), /integer scale/);
});
