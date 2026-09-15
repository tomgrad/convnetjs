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

  const names = ['Net', 'Vol', 'Trainer', 'SGDTrainer', 'MagicNet', 'augment',
    'img_to_vol', 'ConvLayer', 'FullyConnLayer', 'PoolLayer', 'InputLayer',
    'RegressionLayer', 'SoftmaxLayer', 'SVMLayer', 'TanhLayer', 'MaxoutLayer',
    'ReluLayer', 'SigmoidLayer', 'DropoutLayer',
    'LocalResponseNormalizationLayer', 'randf', 'randi', 'randn', 'zeros',
    'maxmin', 'randperm', 'weightedSample', 'arrUnique', 'arrContains',
    'getopt', 'assert'];
  for (const name of names) {
    assert.ok(sandbox.convnetjs[name] !== undefined, 'missing public export: ' + name);
  }
  assert.strictEqual(sandbox.convnetjs.REVISION, 'ALPHA');
});
