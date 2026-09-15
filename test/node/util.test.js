'use strict';
import test from 'node:test';
import assert from 'node:assert/strict';
import convnetjs from '../../build/convnet.cjs';

test('weightedSample never returns undefined for a non-empty list', () => {
  for (let i = 0; i < 10000; i++) {
    // probabilities sum to 0.5, so p frequently exceeds the cumulative sum
    const picked = convnetjs.weightedSample(['a', 'b'], [0.5]);
    assert.notStrictEqual(picked, undefined);
  }
});
