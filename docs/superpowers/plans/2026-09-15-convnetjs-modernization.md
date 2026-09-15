# ConvNetJS Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `src/` from concatenated global-attaching scripts to ES modules with ES classes and modern syntax, and replace the `Makefile` concatenation with an esbuild bundling pipeline, preserving behavior and the `convnetjs` public API.

**Architecture:** Two phases. Phase 1 adds a headless safety net (golden regression + gradient checks) against the current build. Phase 2 first converts every `src/` file to ESM boundaries and switches the build to esbuild (keeping the old syntax so the change is mechanical), then converts each file group's internals to classes/`const`/arrows. The IIFE bundle keeps the global `convnetjs` for demos; a CommonJS bundle feeds the Node tests.

**Tech Stack:** ES modules, ES classes, `esbuild` (devDependency), Node's built-in `node:test`, GNU make is removed.

## Global Constraints

- **Behavior-preserving.** Forward/backward outputs, gradients, loss values, `layer_type` strings, serialization shape, and the `getParamsAndGrads` contract must be identical for identical inputs/weights. Do not change numeric formulas or index arithmetic.
- **API-preserving.** The global `convnetjs` object must expose all 31 public names plus `REVISION` listed in Task 3. Demos and `build/deepqlearn.js|util.js|vis.js` must keep working unchanged.
- **Demos unchanged.** Classic `<script src="../build/convnet.js">` and global `convnetjs`. No `<script type="module">`.
- **No runtime dependencies.** `esbuild` is a devDependency only.
- **Node >= 18.**
- **One commit per task.**
- After Task 3, `build/convnet.js` is the IIFE global build, `build/convnet.cjs` is the CommonJS build, and Node tests import `build/convnet.cjs`.
- The conversion recipe in this plan applies to Tasks 4–8. Read it before each conversion task.

## Conversion Recipe (used by Tasks 4–8)

Apply mechanically; do not restructure files or rename classes.

**1. Constructor → class**
```js
// before
var PoolLayer = function(opt) {
  var opt = opt || {};
  this.sx = opt.sx;
  this.layer_type = 'pool';
}
PoolLayer.prototype = {
  forward: function(V, is_training) { /* ... */ },
  backward: function() { /* ... */ }
}
// after
class PoolLayer {
  constructor(opt = {}) {
    this.sx = opt.sx;
    this.layer_type = 'pool';
  }
  forward(V, is_training) { /* ... */ }
  backward() { /* ... */ }
}
```
- Constructor body moves into `constructor(...)`; `var opt = opt || {}` becomes a default parameter.
- Each `name: function(args) { body }` prototype entry becomes a class method `name(args) { body }`.
- Delete the `X.prototype = { ... }` wrapper.
- Method bodies are copied verbatim except for the general rules below.

**2. Variables and loops:** `var` → `const` when never reassigned, `let` when reassigned (e.g. loop counters, accumulators, module-level mutable state). `for(var i=0;...)` → `for(let i=0;...)`.

**3. Module-local functions:** `var foo = function(a) { ... }` → `function foo(a) { ... }` (or `const foo = (a) => ...` when there is no `this`). Exported util functions stay named exports.

**4. Imports/exports:** already in place from Task 3 — do not add or remove exports; only the syntax changes.

**5. Modern idioms (behavior-neutral only):**
- `Object.prototype.toString.call(x) === '[object Array]'` → `Array.isArray(x)`.
- `typeof x === 'undefined'` → `x === undefined` when `x` is a parameter/local (not a possibly-undeclared global).
- `typeof opt.x !== 'undefined' ? opt.x : d` → keep as-is when `d` may be `0`/`false`; prefer `??` only if the original did not treat `null` specially (these values are never `null`, so `opt.x ?? d` is acceptable).
- `"use strict"` is gone; modules are strict.

**6. Do not touch:** numeric expressions, index arithmetic (`((sx*y)+x)*depth+d`), `Math.pow`/`Math.exp`/`Math.log` calls, the `tanh` saturation helper's thresholds, `this.S_cache_` naming, the `switches`/`switchx`/`switchy` buffers, and the `layer_type` strings.

---

### Task 1: Golden regression tests (FC + serialization)

**Files:**
- Create: `test/node/golden.test.js`

**Interfaces:**
- Consumes: the current CJS build `build/convnet.js` (tests are CommonJS until Task 3).
- Produces: exact expected values that must remain unchanged after the modernization.

- [ ] **Step 1: Write the golden test**

Create `test/node/golden.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

// Layers: 0=input, 1=fc, 2=relu, 3=fc (added by softmax), 4=softmax.
// All weights are pinned so every value below is analytically known.
function makePinnedNet() {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ]);
  net.layers[1].filters[0].w.set([1, 0]); net.layers[1].biases.w[0] = 0.5;
  net.layers[1].filters[1].w.set([0, 1]); net.layers[1].biases.w[1] = -0.25;
  net.layers[1].filters[2].w.set([1, 1]); net.layers[1].biases.w[2] = 0.0;
  net.layers[3].filters[0].w.set([1, 0, 1]); net.layers[3].biases.w[0] = 0.0;
  net.layers[3].filters[1].w.set([0, 1, 1]); net.layers[3].biases.w[1] = 0.0;
  return net;
}

test('golden: forward, backward and gradients are unchanged', () => {
  const net = makePinnedNet();
  const x = new convnetjs.Vol([0.5, -0.25]);

  const out = net.forward(x);
  assert.ok(Math.abs(out.w[0] - 0.7310585786300049) < 1e-12);
  assert.ok(Math.abs(out.w[1] - 0.2689414213699951) < 1e-12);

  const loss = net.backward(1);
  assert.ok(Math.abs(loss - 1.3132616875182228) < 1e-12);

  // gradient at the input
  assert.ok(Math.abs(x.dw[0] - 0.7310585786300049) < 1e-12);
  assert.ok(Math.abs(x.dw[1] - 0) < 1e-12);

  // first fc layer parameter gradients
  assert.deepStrictEqual(Array.from(net.layers[1].filters[0].dw),
    [0.36552928931500245, -0.18276464465750122]);
  assert.ok(Math.abs(net.layers[1].biases.dw[0] - 0.7310585786300049) < 1e-12);
  assert.deepStrictEqual(Array.from(net.layers[1].filters[1].dw), [0, 0]);
  assert.deepStrictEqual(Array.from(net.layers[1].filters[2].dw), [0, 0]);

  // second fc layer parameter gradients
  assert.deepStrictEqual(Array.from(net.layers[3].filters[0].dw),
    [0.7310585786300049, 0, 0.18276464465750122]);
  assert.deepStrictEqual(Array.from(net.layers[3].filters[1].dw),
    [-0.7310585786300049, 0, -0.18276464465750122]);
});

test('golden: serialization shape and round-trip are unchanged', () => {
  const net = makePinnedNet();
  const json = JSON.parse(JSON.stringify(net.toJSON()));

  assert.strictEqual(json.layers.length, 5);
  assert.strictEqual(json.layers.map((l) => l.layer_type).join(','),
    'input,fc,relu,fc,softmax');
  assert.ok(Array.isArray(json.layers[1].filters[0].w));
  assert.deepStrictEqual(json.layers[1].filters[0].w, [1, 0]);

  const restored = new convnetjs.Net();
  restored.fromJSON(json);
  const out = restored.forward(new convnetjs.Vol([0.5, -0.25]));
  assert.ok(Math.abs(out.w[0] - 0.7310585786300049) < 1e-12);
});

test('golden: the global namespace exposes the public API', () => {
  const names = ['Net', 'Vol', 'Trainer', 'SGDTrainer', 'MagicNet', 'augment',
    'img_to_vol', 'ConvLayer', 'FullyConnLayer', 'PoolLayer', 'InputLayer',
    'RegressionLayer', 'SoftmaxLayer', 'SVMLayer', 'TanhLayer', 'MaxoutLayer',
    'ReluLayer', 'SigmoidLayer', 'DropoutLayer',
    'LocalResponseNormalizationLayer', 'randf', 'randi', 'randn', 'zeros',
    'maxmin', 'randperm', 'weightedSample', 'arrUnique', 'arrContains',
    'getopt', 'assert'];
  for (const name of names) {
    assert.ok(convnetjs[name] !== undefined, 'missing public export: ' + name);
  }
});
```

- [ ] **Step 2: Run it to verify it passes against the current build**

Run: `make build/convnet.js && node --test test/node/golden.test.js`

Expected: PASS — 3 tests, 0 fail.

- [ ] **Step 3: Commit**

```bash
git add test/node/golden.test.js
git commit -m "test: add golden regression tests for modernization"
```

---

### Task 2: Headless gradient checks

**Files:**
- Create: `test/node/gradient.test.js`

**Interfaces:**
- Consumes: the current CJS build `build/convnet.js`.
- Produces: numeric-vs-analytic gradient coverage for FC, conv/pool/relu, and maxout, replacing the browser-only Jasmine check.

- [ ] **Step 1: Write the gradient test**

Create `test/node/gradient.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

function randomVol(sx, sy, depth) {
  const v = new convnetjs.Vol(sx, sy, depth, 0.0);
  for (let i = 0; i < v.w.length; i++) { v.w[i] = Math.random() * 2 - 1; }
  return v;
}

function worstRelativeGradientError(makeNet, makeX, label) {
  const net = makeNet();
  const trainer = new convnetjs.SGDTrainer(net,
    { learning_rate: 0.0001, momentum: 0.0, batch_size: 1, l2_decay: 0.0 });
  const x = makeX();
  trainer.train(x, label);

  const delta = 1e-6;
  let worst = 0;
  for (let i = 0; i < x.w.length; i++) {
    const analytic = x.dw[i];
    const old = x.w[i];
    x.w[i] = old + delta; const c0 = net.getCostLoss(x, label);
    x.w[i] = old - delta; const c1 = net.getCostLoss(x, label);
    x.w[i] = old;
    const numeric = (c0 - c1) / (2 * delta);
    const rel = Math.abs(analytic - numeric) / Math.abs(analytic + numeric + 1e-12);
    if (rel > worst) { worst = rel; }
  }
  return worst;
}

test('gradient check: fully connected', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
      { type: 'fc', num_neurons: 5, activation: 'tanh' },
      { type: 'fc', num_neurons: 5, activation: 'tanh' },
      { type: 'softmax', num_classes: 3 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 2), 1) < 1e-2);
});

test('gradient check: conv + pool + relu', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 4, out_sy: 4, out_depth: 1 },
      { type: 'conv', sx: 3, filters: 2, stride: 1, pad: 1, activation: 'relu' },
      { type: 'pool', sx: 2, stride: 2 },
      { type: 'softmax', num_classes: 2 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(4, 4, 1), 0) < 1e-2);
});

test('gradient check: maxout', () => {
  const makeNet = () => {
    const net = new convnetjs.Net();
    net.makeLayers([
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 4 },
      { type: 'fc', num_neurons: 4, activation: 'maxout', group_size: 2 },
      { type: 'softmax', num_classes: 2 }
    ]);
    return net;
  };
  assert.ok(worstRelativeGradientError(makeNet, () => randomVol(1, 1, 4), 0) < 1e-2);
});
```

- [ ] **Step 2: Run it to verify it passes**

Run: `make build/convnet.js && node --test test/node/gradient.test.js`

Expected: PASS — 3 tests, 0 fail (all worst errors < 1e-2; observed ~1e-4 to ~1.3e-3).

- [ ] **Step 3: Run the full suite**

Run: `make build/convnet.js && node --test test/node/`

Expected: all tests pass, 0 fail.

- [ ] **Step 4: Commit**

```bash
git add test/node/gradient.test.js
git commit -m "test: add headless gradient checks for fc, conv and maxout"
```

---

### Task 3: ESM module boundaries + esbuild build switch

This is the one atomic task: the build cannot be green until every `src/` file is a module. Keep the existing syntax (prototypes, `var`); only add imports/exports, remove the IIFE wrappers and `global.` references, and switch the build. Syntax conversion happens in Tasks 4–8.

**Files:**
- Modify: `src/convnet_util.js`, `src/convnet_vol.js`, `src/convnet_vol_util.js`, `src/convnet_layers_dotproducts.js`, `src/convnet_layers_pool.js`, `src/convnet_layers_input.js`, `src/convnet_layers_loss.js`, `src/convnet_layers_nonlinearities.js`, `src/convnet_layers_dropout.js`, `src/convnet_layers_normalization.js`, `src/convnet_net.js`, `src/convnet_trainers.js`, `src/convnet_magicnet.js`
- Delete: `src/convnet_init.js`, `src/convnet_export.js`, `Makefile`
- Create: `src/index.js`
- Modify: `package.json`, `.gitignore`, all files under `test/node/`
- Test: `test/node/*.test.js`

**Interfaces:**
- Consumes: the ESM files produced here.
- Produces: `build/convnet.js` (IIFE global `convnetjs`), `build/convnet.cjs` (CommonJS), `build/convnet-min.js` (minified IIFE). Every later task imports from these modules and must not change the export list.

- [ ] **Step 1: Add the build tooling**

Create `package.json` (replacing its current contents):

```json
{
  "name": "convnetjs",
  "version": "0.0.0",
  "private": true,
  "description": "Deep Learning in Javascript (browser + Node).",
  "type": "module",
  "main": "build/convnet.cjs",
  "exports": {
    ".": {
      "require": "./build/convnet.cjs",
      "default": "./build/convnet.cjs"
    }
  },
  "scripts": {
    "build": "npm run build:iife && npm run build:cjs && npm run build:min",
    "build:iife": "esbuild src/index.js --bundle --format=iife --global-name=convnetjs --outfile=build/convnet.js",
    "build:cjs": "esbuild src/index.js --bundle --format=cjs --outfile=build/convnet.cjs",
    "build:min": "esbuild src/index.js --bundle --format=iife --global-name=convnetjs --minify --outfile=build/convnet-min.js",
    "pretest": "npm run build",
    "test": "node --test test/node/"
  },
  "engines": { "node": ">=18" },
  "devDependencies": { "esbuild": "^0.28.2" },
  "license": "MIT"
}
```

Update `.gitignore` to:

```
build/convnet.js
build/convnet.cjs
build/convnet-min.js
node_modules/
demo/mnist
```

Delete `Makefile`:

```bash
git rm Makefile
```

Install the dev dependency:

```bash
npm install
```

- [ ] **Step 2: Convert every source file's boundaries**

For each file: delete the opening `(function(global) {` and `"use strict";` lines and the trailing `})(convnetjs);`, delete every `global.X = X;` assignment, add the imports listed below at the top, and add the exports listed below at the bottom. Inside method bodies, replace `global.` prefixes with the bare imported name (`global.zeros(...)` → `zeros(...)`, `new global.Vol(...)` → `new Vol(...)`, etc.). Do not otherwise change syntax.

| File | Add imports | Add exports |
|---|---|---|
| `convnet_util.js` | — | `export { randf, randi, randn, zeros, maxmin, randperm, weightedSample, arrUnique, arrContains, getopt, assert };` |
| `convnet_vol.js` | `import { zeros, randn } from './convnet_util.js';` | `export { Vol };` |
| `convnet_vol_util.js` | `import { Vol } from './convnet_vol.js';`<br>`import { randi } from './convnet_util.js';` | `export { augment, img_to_vol };` |
| `convnet_layers_dotproducts.js` | `import { Vol } from './convnet_vol.js';`<br>`import { zeros } from './convnet_util.js';` | `export { ConvLayer, FullyConnLayer };` |
| `convnet_layers_pool.js` | `import { Vol } from './convnet_vol.js';`<br>`import { zeros } from './convnet_util.js';` | `export { PoolLayer };` |
| `convnet_layers_input.js` | `import { getopt } from './convnet_util.js';` | `export { InputLayer };` |
| `convnet_layers_loss.js` | `import { Vol } from './convnet_vol.js';`<br>`import { zeros } from './convnet_util.js';` | `export { RegressionLayer, SoftmaxLayer, SVMLayer };` |
| `convnet_layers_nonlinearities.js` | `import { Vol } from './convnet_vol.js';`<br>`import { zeros } from './convnet_util.js';` | `export { TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer };` |
| `convnet_layers_dropout.js` | `import { zeros } from './convnet_util.js';` | `export { DropoutLayer };` |
| `convnet_layers_normalization.js` | `import { zeros } from './convnet_util.js';` | `export { LocalResponseNormalizationLayer };` |
| `convnet_net.js` | `import { assert } from './convnet_util.js';`<br>`import { ConvLayer, FullyConnLayer } from './convnet_layers_dotproducts.js';`<br>`import { PoolLayer } from './convnet_layers_pool.js';`<br>`import { InputLayer } from './convnet_layers_input.js';`<br>`import { RegressionLayer, SoftmaxLayer, SVMLayer } from './convnet_layers_loss.js';`<br>`import { ReluLayer, SigmoidLayer, TanhLayer, MaxoutLayer } from './convnet_layers_nonlinearities.js';`<br>`import { DropoutLayer } from './convnet_layers_dropout.js';`<br>`import { LocalResponseNormalizationLayer } from './convnet_layers_normalization.js';` | `export { Net };` |
| `convnet_trainers.js` | `import { zeros } from './convnet_util.js';` | `export { Trainer, SGDTrainer };` |
| `convnet_magicnet.js` | `import { randf, randi, maxmin, randperm, weightedSample, getopt, arrUnique } from './convnet_util.js';`<br>`import { Net } from './convnet_net.js';`<br>`import { Trainer } from './convnet_trainers.js';`<br>`import { Vol } from './convnet_vol.js';` | `export { MagicNet };` |

Notes:
- `convnet_net.js` currently has `var assert = global.assert;` — delete it (now imported). Its `makeLayers` switch and `fromJSON` if-chain use `new global.FullyConnLayer` etc.; drop the `global.` prefixes.
- `convnet_magicnet.js` uses `new global.Vol(...)` in `predict_soft`; drop the prefix.
- `convnet_vol_util.js` has no trailing newline; the module rewrite makes that moot.

- [ ] **Step 3: Create `src/index.js`**

```js
import {
  randf, randi, randn, zeros, maxmin, randperm, weightedSample,
  arrUnique, arrContains, getopt, assert
} from './convnet_util.js';
import { Vol } from './convnet_vol.js';
import { augment, img_to_vol } from './convnet_vol_util.js';
import { ConvLayer, FullyConnLayer } from './convnet_layers_dotproducts.js';
import { PoolLayer } from './convnet_layers_pool.js';
import { InputLayer } from './convnet_layers_input.js';
import { RegressionLayer, SoftmaxLayer, SVMLayer } from './convnet_layers_loss.js';
import { TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer } from './convnet_layers_nonlinearities.js';
import { DropoutLayer } from './convnet_layers_dropout.js';
import { LocalResponseNormalizationLayer } from './convnet_layers_normalization.js';
import { Net } from './convnet_net.js';
import { Trainer, SGDTrainer } from './convnet_trainers.js';
import { MagicNet } from './convnet_magicnet.js';

const REVISION = 'ALPHA';

export {
  randf, randi, randn, zeros, maxmin, randperm, weightedSample,
  arrUnique, arrContains, getopt, assert,
  Vol, augment, img_to_vol,
  ConvLayer, FullyConnLayer, PoolLayer, InputLayer,
  RegressionLayer, SoftmaxLayer, SVMLayer,
  TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer,
  DropoutLayer, LocalResponseNormalizationLayer,
  Net, Trainer, SGDTrainer, MagicNet, REVISION
};
```

- [ ] **Step 4: Delete the wrapper files**

```bash
git rm src/convnet_init.js src/convnet_export.js
```

- [ ] **Step 5: Build and check both artifacts**

Run: `npm run build`

Then:

```bash
node -e "const c=require('./build/convnet.cjs'); console.log(typeof c.Net, typeof c.Vol)"
node -e "const fs=require('fs'),vm=require('vm'); const s={}; vm.createContext(s); vm.runInContext(fs.readFileSync('build/convnet.js','utf8'), s); console.log(typeof s.convnetjs.Net, typeof s.convnetjs.randf)"
```

Expected: first line `function function`; second line `function function`.

- [ ] **Step 6: Convert the Node tests to ESM**

In every `test/node/*.test.js`:
- `const test = require('node:test');` → `import test from 'node:test';`
- `const assert = require('node:assert');` → `import assert from 'node:assert/strict';`
- `const convnetjs = require('../../build/convnet.js');` → `import convnetjs from '../../build/convnet.cjs';`

For `test/node/export.test.js`, rewrite it to test the IIFE global build (it replaces the old `convnet_export.js` coverage):

```js
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
```

- [ ] **Step 7: Run the full suite**

Run: `npm test`

Expected: every test passes (smoke, golden, gradient, trainers, nonlinearities, maxout, net, softmax, util, vol, magicnet, export), 0 fail.

- [ ] **Step 8: Commit**

```bash
git add src test package.json .gitignore
git commit -m "refactor: convert src to ES modules and switch build to esbuild"
```

---

### Task 4: Convert `convnet_util.js` to modern syntax

**Files:**
- Modify: `src/convnet_util.js`
- Test: `test/node/` (full suite)

**Interfaces:**
- Consumes: the exports established in Task 3 (unchanged).
- Produces: identical exported function behavior.

- [ ] **Step 1: Confirm baseline green**

Run: `npm test`
Expected: all pass, 0 fail.

- [ ] **Step 2: Apply the conversion recipe**

In `src/convnet_util.js`:
- Convert each `var foo = function(...) {...}` to `function foo(...) {...}` (keep the names and signatures); keep the `export { ... }` statement from Task 3.
- Convert module-level mutable state `var return_v = false; var v_val = 0.0;` to `let return_v = false; let v_val = 0.0;`.
- Inside `gaussRandom`, keep the Box–Muller math exactly; only change `var` → `const`/`let`.
- `zeros`: remove the `typeof ArrayBuffer === 'undefined'` branch and always `return new Float64Array(n);` after the existing `n === undefined || isNaN(n)` guard.
- `arrContains`/`arrUnique`: keep the loop logic; use `const`/`let` and `for (const elt of arr)`.
- `maxmin`, `randperm`, `weightedSample`, `getopt`, `assert`: keep logic identical; `randperm`'s accumulator/swap vars become `let`.
- `assert`: keep the `Error`/fallback behavior.

- [ ] **Step 3: Run the full suite**

Run: `npm test`
Expected: all pass, 0 fail (same count as Task 3).

- [ ] **Step 4: Commit**

```bash
git add src/convnet_util.js
git commit -m "refactor: modernize convnet_util syntax"
```

---

### Task 5: Convert `convnet_vol.js` + `convnet_vol_util.js`

**Files:**
- Modify: `src/convnet_vol.js`, `src/convnet_vol_util.js`
- Test: `test/node/` (full suite)

**Interfaces:**
- Consumes: `Vol` and the util functions.
- Produces: `Vol` as a class with the same constructor overloads and methods; `augment`/`img_to_vol` unchanged.

- [ ] **Step 1: Confirm baseline green**

Run: `npm test`

- [ ] **Step 2: Convert `Vol` to a class**

In `src/convnet_vol.js`, per the recipe:
- `class Vol { constructor(sx, sy, depth, c) { ... } }`, keeping the array-vs-dimensions overload and the weight-normalization init exactly.
- Methods `get`, `set`, `add`, `get_grad`, `set_grad`, `add_grad`, `cloneAndZero`, `clone`, `addFrom`, `addFromScaled`, `setConst`, `toJSON`, `fromJSON` become class methods with bodies copied verbatim.
- Keep `Array.isArray(sx)` in place of the `Object.prototype.toString` check.
- Keep `export { Vol };`.

- [ ] **Step 3: Convert `convnet_vol_util.js`**

- `augment` and `img_to_vol` stay module functions: `function augment(V, crop, dx, dy, fliplr = false) {...}` and `function img_to_vol(img, convert_grayscale = false) {...}`.
- Replace the `if(typeof(fliplr)==='undefined') var fliplr = false;` style defaults with the default parameters shown above.
- Keep the Firefox `NS_ERROR_NOT_AVAILABLE` / `IndexSizeError` handling and the pixel normalization exactly.
- Keep `export { augment, img_to_vol };`.

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: all pass, 0 fail.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_vol.js src/convnet_vol_util.js
git commit -m "refactor: convert Vol and vol utils to classes and modern syntax"
```

---

### Task 6: Convert dotproducts, pool and input layers

**Files:**
- Modify: `src/convnet_layers_dotproducts.js`, `src/convnet_layers_pool.js`, `src/convnet_layers_input.js`
- Test: `test/node/` (full suite)

**Interfaces:**
- Consumes: `Vol`, `zeros`, `getopt`.
- Produces: `ConvLayer`, `FullyConnLayer`, `PoolLayer`, `InputLayer` as classes with identical behavior.

- [ ] **Step 1: Confirm baseline green**

Run: `npm test`

- [ ] **Step 2: Apply the conversion recipe to all three files**

- Convert `ConvLayer`, `FullyConnLayer`, `PoolLayer`, `InputLayer` constructors to classes; move each `X.prototype = { ... }` entry to a method.
- `constructor(opt = {})` replaces `var opt = opt || {}`.
- Keep the optimized conv forward/backward index math byte-for-byte (`V.w[((V_sx * oy)+ox)*V.depth+fd]`, etc.).
- Keep `this.switchx`/`this.switchy` buffers and the pooling window logic unchanged.
- `InputLayer` keeps `getopt(opt, ['out_depth','depth'], 0)`.
- Keep the existing `export` statements.

- [ ] **Step 3: Run the full suite**

Run: `npm test`
Expected: all pass, 0 fail (the conv/pool gradient check in Task 2 covers these).

- [ ] **Step 4: Commit**

```bash
git add src/convnet_layers_dotproducts.js src/convnet_layers_pool.js src/convnet_layers_input.js
git commit -m "refactor: convert dotproduct, pool and input layers to classes"
```

---

### Task 7: Convert loss, nonlinearity, dropout and normalization layers

**Files:**
- Modify: `src/convnet_layers_loss.js`, `src/convnet_layers_nonlinearities.js`, `src/convnet_layers_dropout.js`, `src/convnet_layers_normalization.js`
- Test: `test/node/` (full suite)

**Interfaces:**
- Consumes: `Vol`, `zeros`.
- Produces: `RegressionLayer`, `SoftmaxLayer`, `SVMLayer`, `TanhLayer`, `MaxoutLayer`, `ReluLayer`, `SigmoidLayer`, `DropoutLayer`, `LocalResponseNormalizationLayer` as classes with identical behavior.

- [ ] **Step 1: Confirm baseline green**

Run: `npm test`

- [ ] **Step 2: Apply the conversion recipe**

- Convert every listed layer to a class; move prototype entries to methods.
- Keep the `tanh` helper as a module-local `function tanh(x)` with its `> 20` / `< -20` saturation guards.
- Keep `SoftmaxLayer`'s max-subtraction and `Math.max(this.es[y], 1e-15)` clamp exactly.
- Keep `MaxoutLayer`'s two forward branches and `switches` handling exactly.
- Keep `DropoutLayer`'s train/predict scaling exactly.
- Keep `LocalResponseNormalizationLayer`'s `S_cache_` and `Math.pow` math exactly (do not "fix" it).
- Keep the existing `export` statements.

- [ ] **Step 3: Run the full suite**

Run: `npm test`
Expected: all pass, 0 fail (golden + maxout gradient check cover these).

- [ ] **Step 4: Commit**

```bash
git add src/convnet_layers_loss.js src/convnet_layers_nonlinearities.js src/convnet_layers_dropout.js src/convnet_layers_normalization.js
git commit -m "refactor: convert loss, nonlinearity, dropout and normalization layers to classes"
```

---

### Task 8: Convert net, trainers and magicnet

**Files:**
- Modify: `src/convnet_net.js`, `src/convnet_trainers.js`, `src/convnet_magicnet.js`
- Test: `test/node/` (full suite)

**Interfaces:**
- Consumes: all layer classes, `zeros`, util functions, `Vol`.
- Produces: `Net`, `Trainer`, `SGDTrainer`, `MagicNet` as classes with identical behavior.

- [ ] **Step 1: Confirm baseline green**

Run: `npm test`

- [ ] **Step 2: Apply the conversion recipe**

- `Net`: constructor + methods (`makeLayers`, `forward`, `getCostLoss`, `backward`, `getParamsAndGrads`, `getPrediction`, `toJSON`, `fromJSON`). Move the inner `var desugar = function() {...}` to a method-local `const desugar = () => {...}` (it does not use `this`) or keep it as a local `function desugar()`; behavior must not change. Keep the `makeLayers` switch and the `fromJSON` if-chain with the `Unknown layer type` guard exactly.
- `Trainer`: class with `train`. Keep the Adam/Adagrad/Adadelta/Windowgrad/Nesterov/SGD branches and the `update_count` bias correction exactly. `options` default parameter replaces `var options = options || {}`. Keep `export { Trainer, SGDTrainer };` (SGDTrainer is the same class).
- `MagicNet`: class with `sampleFolds`, `sampleCandidate`, `sampleCandidates`, `step`, `evalValErrors`, `predict_soft`, `predict`, `toJSON`, `fromJSON`, `onFinishFold`, `onFinishBatch`. Keep the `predict_soft` `nv === 0` guard and the `new Vol(1,1,0,0.0)` empty-prediction path.
- Keep the existing `export` statements.

- [ ] **Step 3: Run the full suite**

Run: `npm test`
Expected: all pass, 0 fail (golden serialization and the Adam test cover these).

- [ ] **Step 4: Build and smoke both artifacts**

```bash
npm run build
node -e "const c=require('./build/convnet.cjs'); const n=new c.Net(); n.makeLayers([{type:'input',out_sx:1,out_sy:1,out_depth:2},{type:'fc',num_neurons:3,activation:'relu'},{type:'softmax',num_classes:2}]); console.log(n.forward(new c.Vol([0.1,0.2])).w.length)"
```

Expected: prints `2`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_net.js src/convnet_trainers.js src/convnet_magicnet.js
git commit -m "refactor: convert Net, trainers and MagicNet to classes"
```

---

### Task 9: Update documentation

**Files:**
- Modify: `AGENTS.md`, `Readme.md`

**Interfaces:**
- Consumes: the build/test commands from Task 3.
- Produces: docs that match the new toolchain.

- [ ] **Step 1: Update `AGENTS.md`**

- Replace the `## Build` section (currently describing `make`, `SRCS`, concatenation, and `awk 1`) with:
  - `npm install` once, then `npm run build`.
  - Outputs: `build/convnet.js` (IIFE global `convnetjs`, for demos), `build/convnet.cjs` (CommonJS, for Node/tests), `build/convnet-min.js` (minified IIFE).
  - `src/index.js` is the entry; esbuild resolves the module graph, so there is no manual file order.
  - `build/convnet.js`, `build/convnet.cjs`, `build/convnet-min.js` are generated and not committed.
- Update the `## Tests` section to `npm test` (runs `npm run build` first) and `node --test test/node/<name>.test.js` for a single file; note the Node suite now includes golden and gradient checks.
- Update the `## Layout` notes: `src/*.js` are ES modules; layers are ES classes; `convnet_net.js` still desugars `layer_defs` (activations, implicit `fc` for `softmax`/`svm`/`regression`, dropout); remove the `Makefile`/`SRCS`/`awk` references; keep the note that `build/deepqlearn.js|util.js|vis.js` are separate hand-maintained files.
- Remove the now-obsolete note about `convnet_vol_util.js` having no trailing newline.

- [ ] **Step 2: Update `Readme.md`**

Replace the "Compiling the library from src/ to build/" section with:

```markdown
## Building from src/ to build/
The library is written as ES modules in `src/` and bundled with [esbuild](https://esbuild.github.io/). Install the dev dependency once, then build:

    $ npm install
    $ npm run build

This produces `build/convnet.js` (a global `convnetjs` build used by the demos), `build/convnet.cjs` (CommonJS), and `build/convnet-min.js` (minified).
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md Readme.md
git commit -m "docs: update build and test instructions for the esbuild workflow"
```

---

### Task 10: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Clean build from scratch**

```bash
rm -f build/convnet.js build/convnet.cjs build/convnet-min.js && rm -rf node_modules
npm install
npm run build
```

Note: `build/deepqlearn.js`, `build/util.js`, and `build/vis.js` are tracked hand-maintained files and must not be deleted.

Expected: no errors; `build/convnet.js`, `build/convnet.cjs`, `build/convnet-min.js` exist.

- [ ] **Step 2: Full test suite**

Run: `npm test`
Expected: all tests pass, 0 fail.

- [ ] **Step 3: Artifact checks**

```bash
node -e "const c=require('./build/convnet.cjs'); console.log(typeof c.Net, typeof c.Vol, typeof c.SGDTrainer)"
node -e "const fs=require('fs'),vm=require('vm'); const s={}; vm.createContext(s); vm.runInContext(fs.readFileSync('build/convnet.js','utf8'), s); const need=['Net','Vol','Trainer','SGDTrainer','MagicNet','augment','img_to_vol','ConvLayer','FullyConnLayer','PoolLayer','InputLayer','RegressionLayer','SoftmaxLayer','SVMLayer','TanhLayer','MaxoutLayer','ReluLayer','SigmoidLayer','DropoutLayer','LocalResponseNormalizationLayer','randf','randi','randn','zeros','maxmin','randperm','weightedSample','arrUnique','arrContains','getopt','assert']; const missing=need.filter(n=>s.convnetjs[n]===undefined); console.log('missing:', missing.join(',')||'none')"
```

Expected: `function function function`; `missing: none`.

- [ ] **Step 4: Source hygiene**

```bash
grep -rn "global\.\|(function(global)\|require(" src/ || echo "no legacy patterns"
grep -rn "^var \| var " src/ | grep -v "//" || echo "no var declarations"
```

Expected: no `global.`, no IIFE wrappers, no `require(`; no remaining `var` declarations (comments may still mention them, which is fine).

- [ ] **Step 5: Manual browser checks (cannot be automated here)**

- Open `test/jasmine/SpecRunner.html` in a browser: the gradient spec passes.
- Open `demo/classify2d.html` and `demo/mnist.html`: the page loads, `convnetjs` is defined, and training starts (MNIST requires the images from Task `unpack_mnist.py`).

- [ ] **Step 6: Confirm the working tree is clean**

Run: `git status --short`
Expected: only generated `build/` artifacts (ignored) and `demo/mnist/` (ignored); no source changes uncommitted.

---

## Out of Scope

- TypeScript / `.d.ts` types.
- ESLint / Prettier.
- CI.
- An ESM consumer entry (only the IIFE global and CommonJS builds ship).
- Public API changes or new features.
- Hot-loop performance work.
- Converting demo HTML/JS or the browser Jasmine spec to modules.
- Deleting the stale `bower.json`.
