# ConvNetJS Modernization — Design

Date: 2026-09-15
Status: Approved (pending written-spec review)

## Goal

Modernize the ConvNetJS source and build: convert `src/` from concatenated global-attaching scripts to real ES modules with ES classes and modern syntax, and replace the ad-hoc `Makefile` concatenation with an esbuild bundling pipeline — while preserving the public API and the behavior that the browser demos depend on.

## Decisions (from brainstorming)

- **Scope:** language & modules (not tooling/TS/perf).
- **Compatibility:** old browsers (IE11 / pre-ES2015) may be dropped. Target evergreen browsers + Node 18+.
- **Consumers:** the existing demo HTML loads a global script build and must keep working. A CommonJS build is also produced for the Node test suite.
- **Conversion depth:** ES modules + ES classes, same file layout and behavior (no file reorg/renames beyond adding `index.js` and removing two wrappers).
- **Execution:** single coordinated switch (not a parallel legacy build).
- **Build tool:** `esbuild` as a devDependency, driven by npm scripts; the `Makefile` is deleted.

## Global constraints

- **Behavior-preserving.** Forward/backward outputs, gradients, serialization format, `layer_type` strings, and the `getParamsAndGrads` contract must be byte-for-byte identical to the current implementation for the same inputs and weights.
- **API-preserving.** The global `convnetjs` object must expose the same 29 names it does today, so demos and the hand-maintained `build/deepqlearn.js`, `build/util.js`, `build/vis.js` keep working unchanged.
- **Demos unchanged.** Demo HTML continues to use `<script src="../build/convnet.js">` and the global `convnetjs`; no `<script type="module">` migration.
- **No new runtime dependencies.** esbuild is a devDependency only.
- **Node >= 18** for building and testing.

## 1. Source module graph (`src/`)

Keep the existing file names and responsibilities. Delete `convnet_init.js` (global declaration) and `convnet_export.js` (Node/global export shim); their roles move to the module exports and the bundler.

| File | Imports | Exports |
|---|---|---|
| `convnet_util.js` | — | `randf, randi, randn, zeros, maxmin, randperm, weightedSample, arrUnique, arrContains, getopt, assert` |
| `convnet_vol.js` | `zeros, randn` | `Vol` |
| `convnet_vol_util.js` | `Vol`, `randi` | `augment, img_to_vol` |
| `convnet_layers_dotproducts.js` | `Vol` | `ConvLayer, FullyConnLayer` |
| `convnet_layers_pool.js` | `Vol, zeros` | `PoolLayer` |
| `convnet_layers_input.js` | `getopt` | `InputLayer` |
| `convnet_layers_loss.js` | `Vol, zeros` | `RegressionLayer, SoftmaxLayer, SVMLayer` |
| `convnet_layers_nonlinearities.js` | `Vol, zeros` | `TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer` |
| `convnet_layers_dropout.js` | `zeros` | `DropoutLayer` |
| `convnet_layers_normalization.js` | `zeros` | `LocalResponseNormalizationLayer` |
| `convnet_net.js` | `assert` + all layer classes | `Net` |
| `convnet_trainers.js` | `zeros` | `Trainer, SGDTrainer` |
| `convnet_magicnet.js` | `randf, randi, Net, Trainer, maxmin, randperm, weightedSample, getopt, arrUnique, Vol` | `MagicNet` |
| `index.js` | all of the above | full public surface + `REVISION` |

`index.js` exports exactly the names currently attached to `global`:

```
randf, randi, randn, zeros, maxmin, randperm, weightedSample, arrUnique,
arrContains, getopt, assert,
Vol, augment, img_to_vol,
ConvLayer, FullyConnLayer, PoolLayer, InputLayer, RegressionLayer,
SoftmaxLayer, SVMLayer, TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer,
DropoutLayer, LocalResponseNormalizationLayer,
Net, Trainer, SGDTrainer, MagicNet, REVISION
```

`REVISION` keeps the current value (`'ALPHA'`).

## 2. Syntax conversion

- Prototype-object constructors (`var X = function(){}; X.prototype = {...}`) → `class X { constructor(){} method(){} }`, preserving method bodies and names.
- `var` → `const`/`let`; arrow functions for local helpers where `this` is not used (e.g. util functions, `desugar`).
- Remove the `(function(global){ "use strict"; ... })(convnetjs)` IIFE wrappers (modules are strict by default).
- Replace every `global.X` reference with an explicit import.
- Modernize small idioms without changing behavior:
  - `Object.prototype.toString.call(x) === '[object Array]'` → `Array.isArray(x)`.
  - `typeof x === 'undefined'` → `x === undefined` where safe.
  - Default parameters for option objects (e.g. `constructor(opt = {})`).
  - Remove the `ArrayBuffer` fallback in `zeros`; always return `new Float64Array(n)`.
- Keep module-local helpers (e.g. the `tanh` saturation helper) as non-exported functions.
- Do not change numeric formulas, index arithmetic, or the loss/gradient math.

## 3. Build & distribution

Add `esbuild` as a devDependency and drive it from npm scripts:

- `build:iife` → `esbuild src/index.js --bundle --format=iife --global-name=convnetjs --outfile=build/convnet.js` (demos)
- `build:cjs` → `esbuild src/index.js --bundle --format=cjs --outfile=build/convnet.cjs` (Node/tests)
- `build:min` → `esbuild src/index.js --bundle --format=iife --global-name=convnetjs --minify --outfile=build/convnet-min.js`
- `build` runs all three; `pretest` runs `build`; `test` runs `node --test test/node/`.

The IIFE bundle must define a top-level `convnetjs` global (esbuild's `--global-name`) so classic `<script>` demos and `build/deepqlearn.js|util.js|vis.js` continue to work.

`package.json` changes:
- `"type": "module"`
- `"main": "build/convnet.cjs"`
- `"exports": { ".": { "require": "./build/convnet.cjs" } }`
- `"devDependencies": { "esbuild": "^0.28.2" }`

Other:
- Delete `Makefile`.
- `.gitignore`: add `build/convnet.cjs` (keep `build/convnet.js`, `build/convnet-min.js`, `node_modules/`).

## 4. Tests & verification

Phase 1 — safety net, before any conversion:
- Add golden regression tests against the current build: a deterministic net with explicitly set weights, asserting forward outputs, input gradients (`dw`), parameter gradients, and `toJSON` output.
- Port the analytic-vs-numeric gradient check from `test/jasmine/spec/NeuralNetSpec.js` into `test/node/` so it runs headlessly.
- Cover the less-tested layers (conv/pool/relu, dropout, maxout) with at least one golden or gradient assertion each.

Phase 2 — after conversion:
- Convert `test/node/*.test.js` to ESM (`import test from 'node:test'`, `import assert from 'node:assert/strict'`, `import convnetjs from '../../build/convnet.cjs'`).
- Replace `export.test.js` with a test that executes the IIFE `build/convnet.js` in a bare `vm` context and asserts `sandbox.convnetjs.Net` is a function (validates the demo global build).
- All Phase 1 golden tests must pass unchanged after conversion.
- Manual: open `test/jasmine/SpecRunner.html` and at least one demo (`demo/mnist.html`) to confirm the global build still loads.

## 5. Documentation

- `AGENTS.md`: replace the build section (`make`/`SRCS`/concatenation) with the ESM + esbuild workflow (`npm run build`, outputs, `npm test`); update the layout notes to describe modules/classes and `index.js`.
- `Readme.md`: update the "Compiling the library" section to `npm install` + `npm run build`; keep the output filenames.

## 6. Out of scope

- TypeScript / `.d.ts` types.
- ESLint / Prettier and other lint tooling.
- CI setup.
- An ESM consumer build/entry (only the global IIFE and CommonJS builds are produced).
- Public API redesign or new features.
- Hot-loop performance work.
- Converting demo HTML/JS to ES modules.

## Risks

- **Thin test coverage.** The class conversion is broad; the golden tests and ported gradient check are the guard. Mitigation: add the Phase 1 safety net first and require it to pass unchanged after conversion.
- **Global-name mismatch.** If the esbuild global name or export shape differs, demos and `deepqlearn.js`/`vis.js`/`util.js` break. Mitigation: an explicit `vm` test asserting `convnetjs.Net` on the IIFE bundle, plus a manual demo load.
- **`"type": "module"` fallout.** Node treats `test/node/*.js` as ESM; they must be converted. The browser Jasmine spec and `demo/js/*.js` are classic scripts unaffected by package type.
- **Serialization parity.** `Vol.toJSON` now emits arrays; the golden `toJSON` assertion locks the exact shape.
