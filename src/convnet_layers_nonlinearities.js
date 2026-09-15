import { Vol } from './convnet_vol.js';
import { zeros } from './convnet_util.js';

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
class ReluLayer {
  constructor(opt = {}) {
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'relu';
  }

  forward(V, is_training) {
    this.in_act = V;
    const V2 = V.clone();
    const N = V.w.length;
    const V2w = V2.w;
    for(let i=0;i<N;i++) { 
      if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    const V = this.in_act; // we need to set dw of this
    const V2 = this.out_act;
    const N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for(let i=0;i<N;i++) {
      if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
      else V.dw[i] = V2.dw[i];
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }
}

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
class SigmoidLayer {
  constructor(opt = {}) {
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'sigmoid';
  }

  forward(V, is_training) {
    this.in_act = V;
    const V2 = V.cloneAndZero();
    const N = V.w.length;
    const V2w = V2.w;
    const Vw = V.w;
    for(let i=0;i<N;i++) { 
      V2w[i] = 1.0/(1.0+Math.exp(-Vw[i]));
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    const V = this.in_act; // we need to set dw of this
    const V2 = this.out_act;
    const N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for(let i=0;i<N;i++) {
      const v2wi = V2.w[i];
      V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i];
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }
}

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size
class MaxoutLayer {
  constructor(opt = {}) {
    // required
    this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = Math.floor(opt.in_depth / this.group_size);
    this.layer_type = 'maxout';

    this.switches = zeros(this.out_sx*this.out_sy*this.out_depth); // useful for backprop
  }

  forward(V, is_training) {
    this.in_act = V;
    const N = this.out_depth; 
    const V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

    // optimization branch. If we're operating on 1D arrays we dont have
    // to worry about keeping track of x,y,d coordinates inside
    // input volumes. In convnets we do :(
    if(this.out_sx === 1 && this.out_sy === 1) {
      for(let i=0;i<N;i++) {
        const ix = i * this.group_size; // base index offset
        let a = V.w[ix];
        let ai = 0;
        for(let j=1;j<this.group_size;j++) {
          const a2 = V.w[ix+j];
          if(a2 > a) {
            a = a2;
            ai = j;
          }
        }
        V2.w[i] = a;
        this.switches[i] = ix + ai;
      }
    } else {
      let n=0; // counter for switches
      for(let x=0;x<V.sx;x++) {
        for(let y=0;y<V.sy;y++) {
          for(let i=0;i<N;i++) {
            const ix = i * this.group_size;
            let a = V.get(x, y, ix);
            let ai = 0;
            for(let j=1;j<this.group_size;j++) {
              const a2 = V.get(x, y, ix+j);
              if(a2 > a) {
                a = a2;
                ai = j;
              }
            }
            V2.set(x,y,i,a);
            this.switches[n] = ix + ai;
            n++;
          }
        }
      }

    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    const V = this.in_act; // we need to set dw of this
    const V2 = this.out_act;
    const N = this.out_depth;
    V.dw = zeros(V.w.length); // zero out gradient wrt data

    // pass the gradient through the appropriate switch
    if(this.out_sx === 1 && this.out_sy === 1) {
      for(let i=0;i<N;i++) {
        const chain_grad = V2.dw[i];
        V.dw[this.switches[i]] = chain_grad;
      }
    } else {
      // bleh okay, lets do this the hard way
      let n=0; // counter for switches
      for(let x=0;x<V2.sx;x++) {
        for(let y=0;y<V2.sy;y++) {
          for(let i=0;i<N;i++) {
            const chain_grad = V2.get_grad(x,y,i);
            V.set_grad(x,y,this.switches[n],chain_grad);
            n++;
          }
        }
      }
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.group_size = this.group_size;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
    this.group_size = json.group_size;
    this.switches = zeros(this.out_sx*this.out_sy*this.out_depth);
  }
}

// a helper function, since tanh is not universally available in old browsers.
// Saturate beyond +/-20 to avoid Math.exp overflow to Infinity.
function tanh(x) {
  if (x > 20) { return 1; }
  if (x < -20) { return -1; }
  const y = Math.exp(2 * x);
  return (y - 1) / (y + 1);
}
// Implements Tanh nnonlinearity elementwise
// x -> tanh(x) 
// so the output is between -1 and 1.
class TanhLayer {
  constructor(opt = {}) {
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'tanh';
  }

  forward(V, is_training) {
    this.in_act = V;
    const V2 = V.cloneAndZero();
    const N = V.w.length;
    for(let i=0;i<N;i++) { 
      V2.w[i] = tanh(V.w[i]);
    }
    this.out_act = V2;
    return this.out_act;
  }

  backward() {
    const V = this.in_act; // we need to set dw of this
    const V2 = this.out_act;
    const N = V.w.length;
    V.dw = zeros(N); // zero out gradient wrt data
    for(let i=0;i<N;i++) {
      const v2wi = V2.w[i];
      V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }
}

export { TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer };
