import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

test('the IIFE build exposes the global convnetjs object', () => {
  const code = fs.readFileSync(new URL('../../build/convnet.js', import.meta.url), 'utf8');
  const sandbox = {}; // classic script host: no window, no module
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox);
  assert.strictEqual(typeof sandbox.convnetjs, 'object');
  assert.strictEqual(typeof sandbox.convnetjs.Net, 'function');
  assert.strictEqual(typeof sandbox.convnetjs.randf, 'function');
});
