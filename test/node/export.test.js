'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const vm = require('node:vm');

test('the library attaches to globalThis when window and module are absent', () => {
  const code = fs.readFileSync(require.resolve('../../build/convnet.js'), 'utf8');
  const sandbox = {}; // no window, no module
  vm.createContext(sandbox);

  vm.runInContext(code, sandbox);

  assert.strictEqual(typeof sandbox.convnetjs, 'object');
  assert.strictEqual(typeof sandbox.convnetjs.Net, 'function');
});
