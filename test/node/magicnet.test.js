'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('predict returns -1 when no candidates have been evaluated', () => {
  const magic = new convnetjs.MagicNet([], []);
  assert.strictEqual(magic.predict(new convnetjs.Vol([1.0])), -1);
});
