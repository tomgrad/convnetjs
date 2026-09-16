import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('predict returns -1 when no candidates have been evaluated', () => {
  const magic = new convnetjs.MagicNet([], []);
  assert.strictEqual(magic.predict(new convnetjs.Vol([1.0])), -1);
});

function makeSoftmaxNet() {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'softmax', num_classes: 2 }
  ]);
  return net;
}

test('predict_soft averages into a fresh volume without mutating ensemble outputs', () => {
  const a = makeSoftmaxNet();
  const b = makeSoftmaxNet();
  const magic = new convnetjs.MagicNet([], []);
  magic.ensemble_size = 2;
  magic.evaluated_candidates = [{ net: a }, { net: b }];
  const x = new convnetjs.Vol([0.5, -0.5]);

  const result = magic.predict_soft(x);

  // The first network's output volume must be untouched by the averaging.
  const aAfter = Array.from(a.layers[a.layers.length - 1].out_act.w);
  a.forward(x);
  assert.deepStrictEqual(aAfter, Array.from(a.layers[a.layers.length - 1].out_act.w));

  // ...and the result is the mean of the two networks' predictions.
  b.forward(x);
  const expected = (a.layers[a.layers.length - 1].out_act.w[0]
    + b.layers[b.layers.length - 1].out_act.w[0]) / 2;
  assert.ok(Math.abs(result.w[0] - expected) < 1e-12);
});
