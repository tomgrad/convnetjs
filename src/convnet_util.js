// Random number utilities
let return_v = false;
let v_val = 0.0;
function gaussRandom() {
  if(return_v) { 
    return_v = false;
    return v_val; 
  }
  const u = 2*Math.random()-1;
  const v = 2*Math.random()-1;
  const r = u*u + v*v;
  if(r == 0 || r > 1) return gaussRandom();
  const c = Math.sqrt(-2*Math.log(r)/r);
  v_val = v*c; // cache this
  return_v = true;
  return u*c;
}
function randf(a, b) { return Math.random()*(b-a)+a; }
function randi(a, b) { return Math.floor(Math.random()*(b-a)+a); }
function randn(mu, std){ return mu+gaussRandom()*std; }

// Array utilities
function zeros(n) {
  if(n === undefined || isNaN(n)) { return []; }
  return new Float64Array(n);
}

function arrContains(arr, elt) {
  for(let i=0,n=arr.length;i<n;i++) {
    if(arr[i]===elt) return true;
  }
  return false;
}

function arrUnique(arr) {
  const b = [];
  for(const elt of arr) {
    if(!arrContains(b, elt)) {
      b.push(elt);
    }
  }
  return b;
}

// return max and min of a given non-empty array.
function maxmin(w) {
  if(w.length === 0) { return {}; } // ... ;s
  let maxv = w[0];
  let minv = w[0];
  let maxi = 0;
  let mini = 0;
  const n = w.length;
  for(let i=1;i<n;i++) {
    if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
    if(w[i] < minv) { minv = w[i]; mini = i; } 
  }
  return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
}

// create random permutation of numbers, in range [0...n-1]
function randperm(n) {
  let i = n,
      j = 0,
      temp;
  const array = [];
  for(let q=0;q<n;q++)array[q]=q;
  while (i--) {
      j = Math.floor(Math.random() * (i+1));
      temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
  return array;
}

// sample from list lst according to probabilities in list probs
// the two lists are of same size, and probs adds up to 1
function weightedSample(lst, probs) {
  assert(lst.length === probs.length, 'weightedSample: lst and probs must have the same length');
  const p = randf(0, 1.0);
  let cumprob = 0.0;
  for(let k=0,n=lst.length;k<n;k++) {
    cumprob += probs[k];
    if(p < cumprob) { return lst[k]; }
  }
  return lst[lst.length - 1];
}

// syntactic sugar function for getting default parameter values
function getopt(opt, field_name, default_value) {
  if(typeof field_name === 'string') {
    // case of single string
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
  } else {
    // assume we are given a list of string instead
    let ret = default_value;
    for(let i=0;i<field_name.length;i++) {
      const f = field_name[i];
      if (typeof opt[f] !== 'undefined') {
        ret = opt[f]; // overwrite return value
      }
    }
    return ret;
  }
}

function assert(condition, message) {
  if (!condition) {
    message = message || "Assertion failed";
    if (typeof Error !== "undefined") {
      throw new Error(message);
    }
    throw message; // Fallback
  }
}

export { randf, randi, randn, zeros, maxmin, randperm, weightedSample, arrUnique, arrContains, getopt, assert };
