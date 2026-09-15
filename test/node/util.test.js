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
