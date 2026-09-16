import { Vol } from './convnet_vol.js';
import { zeros } from './convnet_util.js';

// Nearest-neighbor upsampling: each input pixel is repeated into a
// scale x scale block, so out_sx = in_sx * scale and out_sy = in_sy * scale.
// Only mode='nearest' is implemented for now.
class UpsampleLayer {
  // opt is undefined when Net.fromJSON reconstructs the layer; fromJSON then
  // fills in every field, so validation is skipped on that path.
  constructor(opt) {
    if(opt === undefined) {
      this.scale = 1;
      this.layer_type = 'upsample';
      return;
    }

    this.scale = opt.scale;
    if(!Number.isInteger(this.scale) || this.scale < 1) {
      throw new Error('UpsampleLayer requires an integer scale >= 1 (got ' + opt.scale + ')');
    }

    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    this.in_depth = opt.in_depth;

    this.out_sx = this.in_sx * this.scale;
    this.out_sy = this.in_sy * this.scale;
    this.out_depth = this.in_depth;
    this.layer_type = 'upsample';
  }

  forward(V, is_training) {
    this.in_act = V;
    const A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

    const scale = this.scale;
    const V_sx = V.sx;
    const depth = V.depth;

    for(let oy=0; oy<this.out_sy; oy++) {
      const iy = Math.floor(oy / scale);
      for(let ox=0; ox<this.out_sx; ox++) {
        const ix = Math.floor(ox / scale);
        const src = ((V_sx * iy) + ix) * depth;
        const dst = ((this.out_sx * oy) + ox) * depth;
        for(let d=0; d<depth; d++) {
          A.w[dst + d] = V.w[src + d];
        }
      }
    }

    this.out_act = A;
    return this.out_act;
  }

  backward() {
    const V = this.in_act;
    V.dw = zeros(V.w.length);

    const scale = this.scale;
    const V_sx = V.sx;
    const depth = V.depth;

    for(let oy=0; oy<this.out_sy; oy++) {
      const iy = Math.floor(oy / scale);
      for(let ox=0; ox<this.out_sx; ox++) {
        const ix = Math.floor(ox / scale);
        const src = ((V_sx * iy) + ix) * depth;
        const dst = ((this.out_sx * oy) + ox) * depth;
        for(let d=0; d<depth; d++) {
          V.dw[src + d] += this.out_act.dw[dst + d];
        }
      }
    }
  }

  getParamsAndGrads() {
    return [];
  }

  toJSON() {
    const json = {};
    json.scale = this.scale;
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
    this.scale = json.scale;
    this.in_sx = json.in_sx;
    this.in_sy = json.in_sy;
    this.in_depth = json.in_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.out_depth = json.out_depth;
    this.layer_type = json.layer_type;
  }
}

export { UpsampleLayer };
