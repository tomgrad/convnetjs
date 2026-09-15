# AGENTS.md

ConvNetJS: a browser/Node neural-network library. Plain JS, no bundler, no module system — all of `src/` is concatenated into one global `convnetjs` object.

## Build (required before demos/tests)
`build/convnet.js` is a generated artifact and is **not committed**. Demos (`demo/*.html`) and the Jasmine spec runner both load `build/convnet.js`, so build it first.

Build with `make` at the repo root. Requires only `node`/`npx`; it concatenates `src/*.js` into `build/convnet.js`, then minifies to `build/convnet-min.js` with `npx esbuild`. `make clean` removes both outputs. Only `convnet.js` is needed by demos/tests.

- `src/*.js` is concatenated **in the exact order of the `SRCS` list in the `Makefile`**; order is significant (later files reference globals defined earlier).
- To add a new source file, add it to `SRCS` in the `Makefile` at the correct position.
- `convnet_vol_util.js` has no trailing newline; the Makefile's `awk 1` compensates, so don't rely on plain `cat` if you script the build yourself.
- `convnet_init.js` declares the `convnetjs` global; `convnet_export.js` must stay last and wires `module.exports` for Node.
- `build/deepqlearn.js`, `build/util.js`, `build/vis.js` are separate hand-maintained files, not produced by the build.

## Tests
Headless (Node >= 18): `npm test` (runs `make` first) or `make && node --test test/node/`. Run one file with `node --test test/node/<name>.test.js`.

Browser Jasmine 2.0.0 remains at `test/jasmine/SpecRunner.html` (open after building). The Node suite covers trainers/layers/serialization; the Jasmine spec contains the analytic-vs-numeric gradient check.

## Layout
- `src/convnet_net.js` — `Net.makeLayers` desugars `layer_defs`: activations become their own layers, `softmax`/`svm`/`regression` implicitly add an `fc` layer, and `drop_prob` inserts a dropout layer. `net.layers.length` is therefore larger than the input defs.
- `src/convnet_trainers.js` — SGD / Adagrad / Adadelta / Adam / windowgrad trainers.
- `src/convnet_magicnet.js` — k-fold model/hyperparameter search wrapper.
- `demo/js/` — demo logic (not built); demos also use vendored jQuery/pica.
- Data is always 3D `Vol` (`sx`,`sy`,`depth`); non-image inputs use `out_sx=out_sy=1`.

## Notes
- The README's `npm install convnetjs` instructions are stale (upstream is unmaintained); the local `package.json` exists only for building and testing, not for publishing.
- Files use the `(function(global){ "use strict"; ... })(convnetjs)` IIFE pattern and attach classes to the shared global.
