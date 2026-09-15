import { zeros, randn } from './convnet_util.js';

  // Vol is the basic building block of all data in a net.
  // it is essentially just a 3D volume of numbers, with a
  // width (sx), height (sy), and depth (depth).
  // it is used to hold data for all filters, all volumes,
  // all weights, and also stores all gradients w.r.t. 
  // the data. c is optionally a value to initialize the volume
  // with. If c is missing, fills the Vol with random numbers.
  class Vol {
    constructor(sx, sy, depth, c) {
      // this is how you check if a variable is an array. Oh, Javascript :)
      if(Array.isArray(sx)) {
        // we were given a list in sx, assume 1D volume and fill it up
        this.sx = 1;
        this.sy = 1;
        this.depth = sx.length;
        // we have to do the following copy because we want to use
        // fast typed arrays, not an ordinary javascript array
        this.w = zeros(this.depth);
        this.dw = zeros(this.depth);
        for(let i=0;i<this.depth;i++) {
          this.w[i] = sx[i];
        }
      } else {
        // we were given dimensions of the vol
        this.sx = sx;
        this.sy = sy;
        this.depth = depth;
        const n = sx*sy*depth;
        this.w = zeros(n);
        this.dw = zeros(n);
        if(c === undefined) {
          // weight normalization is done to equalize the output
          // variance of every neuron, otherwise neurons with a lot
          // of incoming connections have outputs of larger variance
          const scale = Math.sqrt(1.0/(sx*sy*depth));
          for(let i=0;i<n;i++) { 
            this.w[i] = randn(0.0, scale);
          }
        } else {
          for(let i=0;i<n;i++) { 
            this.w[i] = c;
          }
        }
      }
    }

    get(x, y, d) { 
      const ix=((this.sx * y)+x)*this.depth+d;
      return this.w[ix];
    }
    set(x, y, d, v) { 
      const ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] = v; 
    }
    add(x, y, d, v) { 
      const ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] += v; 
    }
    get_grad(x, y, d) { 
      const ix = ((this.sx * y)+x)*this.depth+d;
      return this.dw[ix]; 
    }
    set_grad(x, y, d, v) { 
      const ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] = v; 
    }
    add_grad(x, y, d, v) { 
      const ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] += v; 
    }
    cloneAndZero() { return new Vol(this.sx, this.sy, this.depth, 0.0)}
    clone() {
      const V = new Vol(this.sx, this.sy, this.depth, 0.0);
      const n = this.w.length;
      for(let i=0;i<n;i++) { V.w[i] = this.w[i]; }
      return V;
    }
    addFrom(V) { for(let k=0;k<this.w.length;k++) { this.w[k] += V.w[k]; }}
    addFromScaled(V, a) { for(let k=0;k<this.w.length;k++) { this.w[k] += a*V.w[k]; }}
    setConst(a) { for(let k=0;k<this.w.length;k++) { this.w[k] = a; }}

    toJSON() {
      // todo: we may want to only save d most significant digits to save space
      const json = {}
      json.sx = this.sx; 
      json.sy = this.sy;
      json.depth = this.depth;
      json.w = Array.prototype.slice.call(this.w);
      return json;
      // we wont back up gradients to save space
    }
    fromJSON(json) {
      this.sx = json.sx;
      this.sy = json.sy;
      this.depth = json.depth;

      const n = this.sx*this.sy*this.depth;
      this.w = zeros(n);
      this.dw = zeros(n);
      // copy over the elements.
      for(let i=0;i<n;i++) {
        this.w[i] = json.w[i];
      }
    }
  }

  export { Vol };
