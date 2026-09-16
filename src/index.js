import {
  randf, randi, randn, zeros, maxmin, randperm, weightedSample,
  arrUnique, arrContains, getopt, assert
} from './convnet_util.js';
import { Vol } from './convnet_vol.js';
import { augment, img_to_vol } from './convnet_vol_util.js';
import { ConvLayer, FullyConnLayer } from './convnet_layers_dotproducts.js';
import { PoolLayer } from './convnet_layers_pool.js';
import { InputLayer } from './convnet_layers_input.js';
import { RegressionLayer, SoftmaxLayer, SVMLayer } from './convnet_layers_loss.js';
import { TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer } from './convnet_layers_nonlinearities.js';
import { DropoutLayer } from './convnet_layers_dropout.js';
import { UpsampleLayer } from './convnet_layers_upsample.js';
import { LocalResponseNormalizationLayer } from './convnet_layers_normalization.js';
import { Net } from './convnet_net.js';
import { Trainer, SGDTrainer } from './convnet_trainers.js';
import { MagicNet } from './convnet_magicnet.js';

const REVISION = 'ALPHA';

export {
  randf, randi, randn, zeros, maxmin, randperm, weightedSample,
  arrUnique, arrContains, getopt, assert,
  Vol, augment, img_to_vol,
  ConvLayer, FullyConnLayer, PoolLayer, InputLayer,
  RegressionLayer, SoftmaxLayer, SVMLayer,
  TanhLayer, MaxoutLayer, ReluLayer, SigmoidLayer,
  DropoutLayer, UpsampleLayer, LocalResponseNormalizationLayer,
  Net, Trainer, SGDTrainer, MagicNet, REVISION
};
