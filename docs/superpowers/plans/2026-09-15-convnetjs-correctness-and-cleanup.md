# ConvNetJS Correctness, Cleanup & Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed correctness bugs in `src/`, add a headless Node test harness to lock them in, then perform low-risk cleanup and a small compatibility-safe modernization.

**Architecture:** The library stays exactly as it is structurally — plain-JS IIFE files concatenated by the `Makefile` into a single global `convnetjs` (CommonJS-exported for Node). We add a dependency-free test harness using Node's built-in `node:test` runner, which exercises the already-generated `build/convnet.js`. Each fix is written test-first against that harness.

**Tech Stack:** Plain JavaScript (ES5 syntax in `src/`), GNU `make`, `node`/`npx`, `esbuild` (via `npx`), Node's built-in `node:test` + `node:assert`.

## Global Constraints

- **`src/*.js` must remain ES5-compatible.** No `let`/`const`/`class`/arrow functions/`Object.assign`/template literals/`Math.tanh` in `src/`. The library supports old browsers (note the `ArrayBuffer` fallback in `convnet_util.js`). Tests (`test/node/**`) may use modern syntax because they only run in Node.
- **No new `src/` files.** All source changes go into existing files; the `Makefile` `SRCS` order must not change.
- **`build/convnet.js` is generated and untracked.** Tests depend on it; run `make` before `node --test`.
- **Test runner is Node >= 18.** Use `node:test` and `node:assert` only — do not add npm dependencies.
- **One commit per task.** Use the commit message given in the task.
- **`make` must stay Java-free** (uses `npx esbuild`), per the existing `Makefile`.

---

### Task 1: Headless Node test harness

**Files:**
- Create: `package.json`
- Create: `.gitignore`
- Create: `test/node/smoke.test.js`
- Modify: `AGENTS.md` (Tests section)
- Modify: `Readme.md` (add a Testing section after the compile section)

**Interfaces:**
- Consumes: the generated `build/convnet.js` CommonJS export.
- Produces: `npm test` (runs `make` then `node --test test/node/`), and the `test/node/` directory convention every later task uses.

- [ ] **Step 1: Write the smoke test**

Create `test/node/smoke.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('the built library exposes the public API', () => {
  assert.strictEqual(typeof convnetjs.Net, 'function');
  assert.strictEqual(typeof convnetjs.Vol, 'function');
  assert.strictEqual(typeof convnetjs.Trainer, 'function');

  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ]);
  const out = net.forward(new convnetjs.Vol([0.1, 0.2]));
  assert.strictEqual(out.w.length, 2);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --test test/node/smoke.test.js`

Expected: FAIL — `Cannot find module '../../build/convnet.js'` (the build output is not committed). If `build/convnet.js` already exists from a previous build, temporarily run `make clean` first so the failure is real.

- [ ] **Step 3: Add the build + test wiring**

Create `package.json`:

```json
{
  "name": "convnetjs",
  "version": "0.0.0",
  "private": true,
  "description": "Deep Learning in Javascript (browser + Node).",
  "main": "build/convnet.js",
  "scripts": {
    "build": "make",
    "pretest": "make",
    "test": "node --test test/node/"
  },
  "engines": { "node": ">=18" },
  "license": "MIT"
}
```

Create `.gitignore`:

```
build/convnet.js
build/convnet-min.js
node_modules/
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/smoke.test.js`

Expected: PASS — output ends with `# pass 1` and `# fail 0`.

- [ ] **Step 5: Document the harness**

In `AGENTS.md`, replace the entire `## Tests` section with:

```markdown
## Tests
Headless (Node >= 18): `npm test` (runs `make` first) or `make && node --test test/node/`. Run one file with `node --test test/node/<name>.test.js`.

Browser Jasmine 2.0.0 remains at `test/jasmine/SpecRunner.html` (open after building). The Node suite covers trainers/layers/serialization; the Jasmine spec contains the analytic-vs-numeric gradient check.
```

In `Readme.md`, insert after the "Compiling the library from src/ to build/" section (before `## Use in Node`):

```markdown
## Testing
Requires Node 18+. Build first, then run the headless suite:

    $ npm test

or, without npm:

    $ make && node --test test/node/

A browser-based Jasmine suite (including a numerical gradient check) lives at `test/jasmine/SpecRunner.html`.
```

- [ ] **Step 6: Commit**

```bash
git add package.json .gitignore test/node/smoke.test.js AGENTS.md Readme.md
git commit -m "test: add headless Node test harness"
```

---

### Task 2: Fix Adam bias correction

Adam computes `m̂ = m/(1-βᵏ)`, but `src/convnet_trainers.js:99-100` multiplies by `(1-βᵏ)` instead of dividing. It also uses `this.k` (every `train()` call) as the bias-correction timestep even though updates only happen every `batch_size` calls.

**Files:**
- Modify: `src/convnet_trainers.js`
- Test: `test/node/trainers.test.js`

**Interfaces:**
- Consumes: `convnetjs.Net`, `convnetjs.Vol`, `convnetjs.Trainer` from `build/convnet.js`.
- Produces: nothing new; `Trainer` behavior changes (Adam now matches Kingma & Ba).

- [ ] **Step 1: Write the failing test**

Create `test/node/trainers.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

// Layers after makeLayers: 0=input, 1=fc, 2=fc (added by regression), 3=regression.
// We pin all weights so the first Adam step is analytically known.
function makePinnedNet() {
  const net = new convnetjs.Net();
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 1 },
    { type: 'fc', num_neurons: 1 },
    { type: 'regression', num_neurons: 1 }
  ]);
  net.layers[1].filters[0].w[0] = 0.0;
  net.layers[1].biases.w[0] = 0.0;
  net.layers[2].filters[0].w[0] = 1.0;
  net.layers[2].biases.w[0] = 0.0;
  return net;
}

test('adam takes a first step of roughly learning_rate against a unit gradient', () => {
  const net = makePinnedNet();
  const trainer = new convnetjs.Trainer(net, {
    method: 'adam',
    learning_rate: 0.1,
    batch_size: 1,
    l2_decay: 0.0,
    l1_decay: 0.0
  });

  // x=1, y=1 => network output 0 => gradient -1 for both fc weight and bias.
  // First Adam step = -lr * mhat/(sqrt(vhat)+eps) = -0.1 * (-1)/(1+eps) ~ +0.1
  trainer.train(new convnetjs.Vol([1.0]), [1.0]);

  assert.ok(Math.abs(net.layers[1].filters[0].w[0] - 0.1) < 1e-4,
    'weight step was ' + net.layers[1].filters[0].w[0]);
  assert.ok(Math.abs(net.layers[1].biases.w[0] - 0.1) < 1e-4,
    'bias step was ' + net.layers[1].biases.w[0]);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/trainers.test.js`

Expected: FAIL — weight step is about `1.0`, not `0.1`.

- [ ] **Step 3: Fix the bias correction and timestep**

In `src/convnet_trainers.js`, add an update counter to the constructor next to `this.k = 0;` (around line 22):

```js
    this.k = 0; // iteration counter
    this.update_count = 0; // number of weight updates actually applied (batch counter)
```

Inside `train`, at the top of the `if(this.k % this.batch_size === 0) {` block (around line 52), increment it:

```js
      if(this.k % this.batch_size === 0) {
        this.update_count++;
```

Replace the Adam branch (lines 97-102) with:

```js
              gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
              xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
              var biasCorr1 = gsumi[j] / (1 - Math.pow(this.beta1, this.update_count)); // correct bias first moment estimate
              var biasCorr2 = xsumi[j] / (1 - Math.pow(this.beta2, this.update_count)); // correct bias second moment estimate
              var dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
              p[j] += dx;
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/trainers.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_trainers.js test/node/trainers.test.js
git commit -m "fix: correct Adam bias correction and update timestep"
```

> Note: this changes training behavior for `method:'adam'`; demo results will differ (they were previously wrong).

---

### Task 3: Fix `tanh` overflow to NaN

`src/convnet_layers_nonlinearities.js:229-232` computes `Math.exp(2*x)`, which overflows to `Infinity` for `x` above ~355, producing `(Inf-1)/(Inf+1) === NaN`.

**Files:**
- Modify: `src/convnet_layers_nonlinearities.js`
- Test: `test/node/nonlinearities.test.js`

**Interfaces:**
- Consumes: `convnetjs.TanhLayer`, `convnetjs.Vol`.
- Produces: `TanhLayer.forward` now returns finite values for all inputs.

- [ ] **Step 1: Write the failing test**

Create `test/node/nonlinearities.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

function tanhLayer() {
  return new convnetjs.TanhLayer({ in_sx: 1, in_sy: 1, in_depth: 1 });
}

test('tanh saturates to +1 and -1 without producing NaN', () => {
  const pos = tanhLayer().forward(new convnetjs.Vol([400]));
  const neg = tanhLayer().forward(new convnetjs.Vol([-400]));
  assert.ok(isFinite(pos.w[0]), 'tanh(400) was ' + pos.w[0]);
  assert.ok(isFinite(neg.w[0]), 'tanh(-400) was ' + neg.w[0]);
  assert.ok(Math.abs(pos.w[0] - 1) < 1e-9);
  assert.ok(Math.abs(neg.w[0] + 1) < 1e-9);
});

test('tanh matches the analytic value for small inputs', () => {
  const out = tanhLayer().forward(new convnetjs.Vol([0.5]));
  assert.ok(Math.abs(out.w[0] - Math.tanh(0.5)) < 1e-9);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/nonlinearities.test.js`

Expected: FAIL — `tanh(400)` is `NaN`.

- [ ] **Step 3: Add saturation guards**

In `src/convnet_layers_nonlinearities.js`, replace the `tanh` helper (lines 228-232) with:

```js
  // a helper function, since tanh is not universally available in old browsers.
  // Saturate beyond +/-20 to avoid Math.exp overflow to Infinity.
  function tanh(x) {
    if (x > 20) { return 1; }
    if (x < -20) { return -1; }
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/nonlinearities.test.js`

Expected: PASS — `# pass 2`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_layers_nonlinearities.js test/node/nonlinearities.test.js
git commit -m "fix: prevent tanh overflow to NaN for large inputs"
```

---

### Task 4: Fix `MaxoutLayer.fromJSON` switch buffer size

`src/convnet_layers_nonlinearities.js:224` restores `this.switches` with `zeros(this.group_size)` instead of the full `out_sx*out_sy*out_depth` buffer. Backprop then writes out of range (silently dropped by `Float64Array`), producing wrong gradients after deserialization.

**Files:**
- Modify: `src/convnet_layers_nonlinearities.js`
- Test: `test/node/maxout.test.js`

**Interfaces:**
- Consumes: `convnetjs.MaxoutLayer`.
- Produces: `MaxoutLayer.fromJSON` allocates the correct `switches` length.

- [ ] **Step 1: Write the failing test**

Create `test/node/maxout.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('maxout fromJSON restores a switches buffer sized to the output volume', () => {
  // 2x2 spatial, depth 4, group_size 2 => out_depth 2 => 2*2*2 = 8 switches
  const layer = new convnetjs.MaxoutLayer({ in_sx: 2, in_sy: 2, in_depth: 4, group_size: 2 });
  const json = layer.toJSON();

  const restored = new convnetjs.MaxoutLayer({});
  restored.fromJSON(json);

  assert.strictEqual(restored.switches.length, 8);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/maxout.test.js`

Expected: FAIL — `restored.switches.length` is `2`.

- [ ] **Step 3: Size the buffer from the output volume**

In `src/convnet_layers_nonlinearities.js`, in `MaxoutLayer.fromJSON` (line 224), change:

```js
      this.switches = global.zeros(this.group_size);
```

to:

```js
      this.switches = global.zeros(this.out_sx*this.out_sy*this.out_depth);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/maxout.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_layers_nonlinearities.js test/node/maxout.test.js
git commit -m "fix: restore maxout switch buffer at correct size"
```

---

### Task 5: Stop `makeLayers` mutating caller `layer_defs`

`src/convnet_net.js` writes `bias_pref`, `in_sx`, `in_sy`, and `in_depth` onto the caller's definition objects. `MagicNet` reuses the same `layer_defs` array across folds, so this is a latent shared-state hazard.

**Files:**
- Modify: `src/convnet_net.js`
- Test: `test/node/net.test.js`

**Interfaces:**
- Consumes: `convnetjs.Net`.
- Produces: `makeLayers` leaves the passed-in array and its objects unchanged.

- [ ] **Step 1: Write the failing test**

Create `test/node/net.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('makeLayers does not mutate the caller layer definitions', () => {
  const defs = [
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 3, activation: 'relu' },
    { type: 'softmax', num_classes: 2 }
  ];
  const before = JSON.stringify(defs);

  const net = new convnetjs.Net();
  net.makeLayers(defs);

  assert.strictEqual(JSON.stringify(defs), before);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/net.test.js`

Expected: FAIL — the definitions gained `bias_pref`/`in_sx`/`in_sy`/`in_depth`.

- [ ] **Step 3: Clone definitions before desugaring**

In `src/convnet_net.js`, immediately before the `// desugar layer_defs ...` comment (line 21), add a shallow clone (manual, to stay ES5):

```js
      // work on copies so we never mutate the caller's layer definitions
      var cloned_defs = [];
      for(var di=0;di<defs.length;di++) {
        var src = defs[di];
        var copy = {};
        for(var key in src) { copy[key] = src[key]; }
        cloned_defs.push(copy);
      }
      defs = cloned_defs;
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/net.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_net.js test/node/net.test.js
git commit -m "fix: clone layer defs in makeLayers instead of mutating caller input"
```

---

### Task 6: Serialize `Vol.w` as a JSON array

`src/convnet_vol.js:91` assigns the `Float64Array` directly to `json.w`. `JSON.stringify` turns a typed array into `{"0":…,"1":…}`, so serialized networks are environment-dependent and bloated.

**Files:**
- Modify: `src/convnet_vol.js`
- Test: `test/node/vol.test.js`

**Interfaces:**
- Consumes: `convnetjs.Vol`.
- Produces: `Vol.toJSON().w` is a plain array; `Vol.fromJSON` already accepts either form.

- [ ] **Step 1: Write the failing test**

Create `test/node/vol.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/vol.test.js`

Expected: FAIL — `w` is not an array (it stringifies to an object).

- [ ] **Step 3: Convert the buffer to a plain array**

In `src/convnet_vol.js`, in `toJSON` (line 91), change:

```js
      json.w = this.w;
```

to:

```js
      json.w = Array.prototype.slice.call(this.w);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/vol.test.js`

Expected: PASS — `# pass 2`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_vol.js test/node/vol.test.js
git commit -m "fix: serialize Vol weights as a JSON array"
```

---

### Task 7: Fail loudly on unknown layer types in `Net.fromJSON`

`src/convnet_net.js:168-181` uses a chain of `if` statements. An unknown `layer_type` leaves `L` undefined and throws an opaque `TypeError` at `L.fromJSON(...)`.

**Files:**
- Modify: `src/convnet_net.js`
- Test: `test/node/net.test.js`

**Interfaces:**
- Consumes: `convnetjs.Net`.
- Produces: `fromJSON` throws `Error('Unknown layer type: <type>')`.

- [ ] **Step 1: Write the failing test**

Append to `test/node/net.test.js`:

```js
test('fromJSON throws a clear error for an unknown layer type', () => {
  const net = new convnetjs.Net();
  assert.throws(
    () => net.fromJSON({ layers: [{ layer_type: 'bogus' }] }),
    /Unknown layer type: bogus/
  );
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/net.test.js`

Expected: FAIL — the thrown message is a `TypeError` about `fromJSON` of `undefined`, not the expected text.

- [ ] **Step 3: Add the explicit error**

In `src/convnet_net.js`, in `fromJSON`, immediately after the last `if(t==='svm') { ... }` line (line 180), add:

```js
        if(typeof L === 'undefined') { throw new Error('Unknown layer type: ' + t); }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/net.test.js`

Expected: PASS — `# pass 2`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_net.js test/node/net.test.js
git commit -m "fix: throw a clear error for unknown layer types in fromJSON"
```

---

### Task 8: Return a finite softmax loss for saturated outputs

`src/convnet_layers_loss.js:70` returns `-Math.log(this.es[y])`, which is `Infinity` when the true class probability underflows to `0` (e.g. logits `[0, 1000]`).

**Files:**
- Modify: `src/convnet_layers_loss.js`
- Test: `test/node/softmax.test.js`

**Interfaces:**
- Consumes: `convnetjs.SoftmaxLayer`, `convnetjs.Vol`.
- Produces: `SoftmaxLayer.backward` returns a finite loss (clamped at `-log(1e-15)`).

- [ ] **Step 1: Write the failing test**

Create `test/node/softmax.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('softmax loss stays finite when the true class underflows to zero', () => {
  const layer = new convnetjs.SoftmaxLayer({ in_sx: 1, in_sy: 1, in_depth: 2 });
  layer.forward(new convnetjs.Vol([0, 1000])); // P(class 0) underflows to 0
  const loss = layer.backward(0);
  assert.ok(isFinite(loss), 'loss was ' + loss);
  assert.ok(loss > 0);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/softmax.test.js`

Expected: FAIL — `loss was Infinity`.

- [ ] **Step 3: Clamp the probability**

In `src/convnet_layers_loss.js`, in `SoftmaxLayer.backward` (line 70), change:

```js
      return -Math.log(this.es[y]);
```

to:

```js
      return -Math.log(Math.max(this.es[y], 1e-15));
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/softmax.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_layers_loss.js test/node/softmax.test.js
git commit -m "fix: keep softmax loss finite for saturated predictions"
```

---

### Task 9: Make `weightedSample` always return an element

`src/convnet_util.js:88-95` can fall through and return `undefined` when the probabilities do not sum to `1` (float drift or bad input).

**Files:**
- Modify: `src/convnet_util.js`
- Test: `test/node/util.test.js`

**Interfaces:**
- Consumes: `convnetjs.weightedSample`.
- Produces: `weightedSample` never returns `undefined` for a non-empty list.

- [ ] **Step 1: Write the failing test**

Create `test/node/util.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('weightedSample never returns undefined for a non-empty list', () => {
  for (let i = 0; i < 10000; i++) {
    // probabilities sum to 0.5, so p frequently exceeds the cumulative sum
    const picked = convnetjs.weightedSample(['a', 'b'], [0.5]);
    assert.notStrictEqual(picked, undefined);
  }
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/util.test.js`

Expected: FAIL — `picked` is `undefined` for roughly half the samples.

- [ ] **Step 3: Add the fallback return**

In `src/convnet_util.js`, in `weightedSample` (after the `for` loop, before the closing brace at line 95), add:

```js
    return lst[lst.length - 1];
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/util.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_util.js test/node/util.test.js
git commit -m "fix: guarantee weightedSample returns an element"
```

---

### Task 10: Cleanup — unused imports, shadowed params, redeclarations

No behavior change. Removes dead code and lint noise.

**Files:**
- Modify: `src/convnet_net.js`
- Modify: `src/convnet_trainers.js`
- Modify: `src/convnet_layers_input.js`
- Modify: `src/convnet_layers_loss.js`
- Modify: `src/convnet_layers_dropout.js`
- Modify: `src/convnet_layers_normalization.js`
- Modify: `src/convnet_layers_dotproducts.js`
- Modify: `src/convnet_layers_pool.js`
- Modify: `src/convnet_layers_nonlinearities.js`
- Modify: `src/convnet_magicnet.js`

**Interfaces:**
- Consumes: existing public API only.
- Produces: identical behavior; no public API change.

- [ ] **Step 1: Remove unused `Vol` locals**

Delete the `var Vol = global.Vol; // convenience` line from each file where `Vol` is never referenced: `src/convnet_net.js` (line 3), `src/convnet_trainers.js` (line 3), `src/convnet_layers_input.js` (line 4), `src/convnet_layers_loss.js` (line 3), `src/convnet_layers_dropout.js` (line 3), `src/convnet_layers_normalization.js` (line 3).

Do **not** remove it from `convnet_vol_util.js`, `convnet_layers_dotproducts.js`, `convnet_layers_pool.js`, or `convnet_layers_nonlinearities.js` — those use `Vol`.

- [ ] **Step 2: Stop shadowing the `opt` parameter**

In every constructor that shadows its `opt` parameter, replace `var opt = opt || {};` with `opt = opt || {};`. This applies to `convnet_layers_dotproducts.js` (ConvLayer, FullyConnLayer), `convnet_layers_pool.js`, `convnet_layers_input.js`, `convnet_layers_loss.js` (Softmax, Regression, SVM), `convnet_layers_nonlinearities.js` (Relu, Sigmoid, Maxout, Tanh), `convnet_layers_dropout.js`, `convnet_layers_normalization.js`, and `convnet_magicnet.js`.

Verify none remain: `git grep -n "var opt = opt"` should print nothing.

- [ ] **Step 3: Remove duplicate `start`/`end` declarations**

In `src/convnet_trainers.js`, rewrite the timing block at the top of `train` to declare `start`/`end` once (currently `var start`/`var end` each appear twice):

```js
      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      end = new Date().getTime();
      var bwd_time = end - start;
```

- [ ] **Step 4: Fix the latent `typeof` bug in maxout desugaring**

In `src/convnet_net.js:57`, change:

```js
              var gs = def.group_size !== 'undefined' ? def.group_size : 2;
```

to:

```js
              var gs = typeof def.group_size !== 'undefined' ? def.group_size : 2;
```

(This is currently masked by `MaxoutLayer`'s own default; the fix makes the intent explicit.)

- [ ] **Step 5: Run the full suite**

Run: `make && node --test test/node/`

Expected: PASS — all tests pass, `# fail 0`.

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "chore: remove unused imports, shadowed params, and duplicate declarations"
```

---

### Task 11: Make the browser/Node export robust with `globalThis`

`src/convnet_export.js:4` assigns to `window`, which does not exist in web workers or other non-Node hosts.

**Files:**
- Modify: `src/convnet_export.js`
- Test: `test/node/export.test.js`

**Interfaces:**
- Consumes: the generated `build/convnet.js`.
- Produces: the library attaches to `globalThis` when `module` is absent; Node still gets `module.exports`.

- [ ] **Step 1: Write the failing test**

Create `test/node/export.test.js`:

```js
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/export.test.js`

Expected: FAIL — `ReferenceError: window is not defined`.

- [ ] **Step 3: Use `globalThis` with fallbacks**

Replace the contents of `src/convnet_export.js` with:

```js
(function(lib) {
  "use strict";
  if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = lib; // in nodejs
  } else {
    // browser, web worker, or other host
    var g = (typeof globalThis !== "undefined") ? globalThis
          : (typeof window !== "undefined") ? window
          : this;
    g.convnetjs = lib;
  }
})(convnetjs);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/export.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Run the full suite and confirm the Node entrypoint still works**

Run: `make && node --test test/node/ && node -e "console.log(typeof require('./build/convnet.js').Net)"`

Expected: all tests pass; the final line prints `function`.

- [ ] **Step 6: Commit**

```bash
git add src/convnet_export.js test/node/export.test.js
git commit -m "fix: attach library via globalThis when window is unavailable"
```

---

### Task 12: Make `MagicNet.predict` safe with no evaluated candidates

`src/convnet_magicnet.js:277-287` dereferences `xout.w` when `predict_soft` returned `undefined` (no candidates yet, e.g. a `MagicNet` constructed with no data).

**Files:**
- Modify: `src/convnet_magicnet.js`
- Test: `test/node/magicnet.test.js`

**Interfaces:**
- Consumes: `convnetjs.MagicNet`, `convnetjs.Vol`.
- Produces: `predict` returns `-1` instead of throwing when there is nothing to average.

- [ ] **Step 1: Write the failing test**

Create `test/node/magicnet.test.js`:

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const convnetjs = require('../../build/convnet.js');

test('predict returns -1 when no candidates have been evaluated', () => {
  const magic = new convnetjs.MagicNet([], []);
  assert.strictEqual(magic.predict(new convnetjs.Vol([1.0])), -1);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make && node --test test/node/magicnet.test.js`

Expected: FAIL — `TypeError: Cannot read properties of undefined (reading 'w')`.

- [ ] **Step 3: Return an empty volume when there is nothing to average**

In `src/convnet_magicnet.js`, in `predict_soft`, immediately after the `if/else` that sets `nv` and `eval_candidates` (before `// forward nets of all candidates and average the predictions`), add:

```js
      if(nv === 0) { return new global.Vol(1, 1, 0, 0.0); }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make && node --test test/node/magicnet.test.js`

Expected: PASS — `# pass 1`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/convnet_magicnet.js test/node/magicnet.test.js
git commit -m "fix: return an empty prediction instead of throwing in MagicNet.predict"
```

---

## Final Verification

- [ ] Run the complete headless suite from a clean build: `make clean && make && node --test test/node/` — expect all tests pass, `# fail 0`.
- [ ] Confirm the Node entrypoint: `node -e "const c=require('./build/convnet.js'); console.log(typeof c.Net, typeof c.Vol)"` — expect `function function`.
- [ ] Open `test/jasmine/SpecRunner.html` in a browser and confirm the existing gradient check still passes (manual; requires `build/convnet.js`).
- [ ] Confirm `src/` contains no ES6 syntax: `git grep -nE "\b(let|const|class)\b|=>|Math\.tanh|Object\.assign" -- src/` prints nothing.
- [ ] Confirm no leftover shadowed params: `git grep -n "var opt = opt"` prints nothing.

---

## Out of Scope (deliberately deferred)

These were identified during analysis but are not included, either because they cannot be tested headlessly or because they carry a compatibility/design decision:

- **`img_to_vol` / `augment` DOM changes** (`src/convnet_vol_util.js`): browser-only (`document`), needs a jsdom or manual browser test; also the plain-array `x.w = pv` assignment.
- **Pooling out-of-bounds switch (`-1`)** (`src/convnet_layers_pool.js`): on `Float64Array` the negative-index write is silently ignored, so it is defensive rather than a reproducible failure.
- **LRN gradient correctness** (`src/convnet_layers_normalization.js`): the author flags it as unverified; fixing needs a numeric gradient harness.
- **Hot-loop micro-optimization** (replace `Vol.get`/`set` calls in pool/maxout/LRN with direct typed-array indexing, as conv/fc already do): a performance change, not a correctness one; needs benchmarks to justify.
- **ES6/ESM rewrite, TypeScript types, bundler, ESLint/Prettier**: breaks the stated legacy-browser support and/or adds tooling dependencies. Decide separately.
- **`build/deepqlearn.js`, `build/util.js`, `build/vis.js`**: hand-maintained, not produced by the build; out of scope for this plan.
