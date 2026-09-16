# AGENTS.md

ConvNetJS: a browser/Node neural-network library. `src/` holds ES modules that esbuild bundles into a global `convnetjs` object (and a CommonJS build).

## Build (required before demos/tests)
`build/convnet.js` is a generated artifact and is **not committed**. Demos (`demo/*.html`) and the Jasmine spec runner both load `build/convnet.js`, so build it first.

Install the dev dependency once, then build:

    $ npm install
    $ npm run build

This produces four generated, uncommitted outputs (each with a `.map` sourcemap):

- `build/convnet.js` — IIFE bundle exposing the global `convnetjs` (used by demos).
- `build/convnet.cjs` — CommonJS bundle (used by Node/tests).
- `build/convnet.mjs` — ESM bundle (used by `import` consumers; the `import` condition in `package.json`).
- `build/convnet-min.js` — minified IIFE bundle.

`src/index.js` is the bundle entry; esbuild resolves the module graph, so there is no manual file order. To add a source file, import it from `src/index.js`.

- `build/deepqlearn.js`, `build/util.js`, `build/vis.js` are separate hand-maintained files, not produced by the build.

## Tests
Headless (Node >= 18): `npm test` (runs `npm run build` first) or `npm run build && node --test test/node/`. Run one file with `node --test test/node/<name>.test.js`.

Browser Jasmine 2.0.0 remains at `test/jasmine/SpecRunner.html` (open after building). The Node suite now includes golden regression tests and headless gradient checks (fc, conv/pool/relu, maxout), plus the original layer/trainer/serialization tests.

## Layout
- `src/*.js` are ES modules; layers are ES classes.
- `src/convnet_net.js` — `Net.makeLayers` desugars `layer_defs`: activations become their own layers, `softmax`/`svm`/`regression` implicitly add an `fc` layer, and `drop_prob` inserts a dropout layer. `net.layers.length` is therefore larger than the input defs.
- `src/convnet_trainers.js` — SGD / Adagrad / Adadelta / Adam / windowgrad trainers.
- `src/convnet_magicnet.js` — k-fold model/hyperparameter search wrapper.
- `demo/js/` — demo logic (not built); demos also use vendored jQuery/pica.
- Data is always 3D `Vol` (`sx`,`sy`,`depth`); non-image inputs use `out_sx=out_sy=1`.

## Notes
- The README's `npm install convnetjs` instructions are stale (upstream is unmaintained); the local `package.json` exists only for building and testing, not for publishing.
- Source files use `import`/`export`; the IIFE bundle attaches the exports to the shared `convnetjs` global.
