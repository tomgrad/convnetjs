import { Vol } from './convnet_vol.js';
import { zeros } from './convnet_util.js';

// Reinterprets a Vol as a new sx x sy x depth volume with the same number of
// elements. It does not resample or reorder: the flat buffer is copied
// unchanged, so sx*sy*depth must equal the input element count.
class ReshapeLayer {
  // opt is undefined when Net.fromJSON reconstructs the layer; fromJSON then
  // fills in every field, so validation is skipped on that path.
  constructor(opt) {
    if(opt === undefined) {
      this.layer_type = 'reshape';
      return;
    }

    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    this.in_depth = opt.in_depth;
    this.out_sx = opt.sx;
    this.out_sy = opt.sy;
    this.out_depth = opt.depth;
    this.layer_type = 'reshape';

    if(!isPositiveInt(this.out_sx) || !isPositiveInt(this.out_sy) || !isPositiveInt(this.out_depth)) {
      throw new Error('ReshapeLayer requires positive integer sx, sy and depth (got ' +
        opt.sx + ', ' + opt.sy + ', ' + opt.depth + ')');
    }

    const in_count = this.in_sx * this.in_sy * this.in_depth;
    const out_count = this.out_sx * this.out_sy * this.out_depth;
    if(in_count !== out_count) {
      throw new Error('ReshapeLayer cannot reshape ' +
        this.in_sx + 'x' + this.in_sy + 'x' + this.in_depth + ' (' + in_count + ' elements) into ' +
        this.out_sx + 'x' + this.out_sy + 'x' + this.out_depth + ' (' + out_count + ' elements)');
    }
  }

  forward(V, is_training) {
    this.in_act = V;
    const A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
    const n = V.w.length;
    for(let i=0;i<n;i++) { A.w[i] = V.w[i]; }
    this.out_act = A;
    return this.out_act;
  }

  backward() {
    const V = this.in_act;
    V.dw = zeros(V.w.length);
    const n = V.w.length;
    for(let i=0;i<n;i++) { V.dw[i] = this.out_act.dw[i]; }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.in_sx = this.in_sx;
    json.in_sy = this.in_sy;
    json.in_depth = this.in_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.out_depth = this.out_depth;
    json.layer_type = this.layer_type;
    return json;
  }

  fromJSON(json) {
    this.in_sx = json.in_sx;
    this.in_sy = json.in_sy;
    this.in_depth = json.in_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.out_depth = json.out_depth;
    this.layer_type = json.layer_type;
  }
}

function isPositiveInt(x) {
  return Number.isInteger(x) && x >= 1;
}

export { ReshapeLayer };
