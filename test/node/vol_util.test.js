import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

function withFakeCanvas(imgData, fn) {
  const ctx = {
    drawImage() {},
    getImageData() { return { data: imgData }; }
  };
  const canvas = { width: 0, height: 0, getContext: () => ctx };
  const original = globalThis.document;
  globalThis.document = { createElement: () => canvas };
  try {
    return fn();
  } finally {
    if (original === undefined) { delete globalThis.document; }
    else { globalThis.document = original; }
  }
}

test('img_to_vol returns a Vol whose w is a Float64Array', () => {
  // 2 pixels, RGBA
  const data = [255, 0, 0, 255, 0, 255, 0, 255];
  const x = withFakeCanvas(data, () => convnetjs.img_to_vol({ width: 2, height: 1 }));

  assert.ok(x.w instanceof Float64Array, 'w was ' + x.w.constructor.name);
  assert.strictEqual(x.sx, 2);
  assert.strictEqual(x.depth, 4);
  assert.deepStrictEqual(Array.from(x.w), [0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]);
});
