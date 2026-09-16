// ConvNetJS layer playground.
// Forward a random MNIST image through a user-defined network and visualize
// the last layer's output as an auto-scaled heatmap.

var MNIST_DIM = 28;
var MNIST_PIXELS = MNIST_DIM * MNIST_DIM;
var MNIST_SAMPLES = 3000;

var img_data = null;   // ImageData of the loaded MNIST batch
var net = null;        // current Net, or null if the last build failed
var current_k = 0;     // index of the current MNIST sample
var current_x = null;  // current input Vol
var layer_defs;        // populated by eval() of the text area

var default_layerdefs = "\
layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:28, out_sy:28, out_depth:1});\n\
layer_defs.push({type:'conv', sx:3, filters:4, stride:1, pad:1, activation:'relu'});\n\
";

function setStatus(msg, isError) {
  var elt = document.getElementById('status');
  elt.textContent = msg;
  elt.style.color = isError ? '#b00' : '#333';
}

function loadData() {
  var img = new Image();
  img.onload = function() {
    var canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setStatus('Loaded a ' + img.width + 'x' + img.height + ' MNIST batch.');
    buildAndRun();
  };
  img.onerror = function() {
    setStatus('Could not load mnist/mnist_batch_0.png. The MNIST assets are not ' +
      'committed; run demo/mnist/unpack_mnist.py first.', true);
  };
  img.src = 'mnist/mnist_batch_0.png';
}

// Reconstruct sample k as a 28x28x1 Vol, matching images-demo.js exactly.
function sampleToVol(k) {
  var p = img_data.data;
  var x = new convnetjs.Vol(MNIST_DIM, MNIST_DIM, 1, 0.0);
  var i = 0;
  for(var xc=0; xc<MNIST_DIM; xc++) {
    for(var yc=0; yc<MNIST_DIM; yc++) {
      var ix = ((MNIST_PIXELS * k) + i) * 4;
      x.set(yc, xc, 0, p[ix]/255.0 - 0.5);
      i++;
    }
  }
  return x;
}

function buildAndRun() {
  try {
    eval(document.getElementById('layerdef').value);
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var L0 = net.layers[0];
    if(L0.layer_type !== 'input' || L0.out_sx !== MNIST_DIM ||
       L0.out_sy !== MNIST_DIM || L0.out_depth !== 1) {
      throw new Error("The first layer must be {type:'input', out_sx:28, out_sy:28, out_depth:1} to match an MNIST image.");
    }

    current_k = Math.floor(Math.random() * MNIST_SAMPLES);
    run();
  } catch(err) {
    net = null;
    setStatus('Error: ' + err.message, true);
  }
}

function newRandomImage() {
  if(!net) { buildAndRun(); return; }
  current_k = Math.floor(Math.random() * MNIST_SAMPLES);
  run();
}

function run() {
  if(!img_data) { setStatus('MNIST data is still loading...', true); return; }
  try {
    current_x = sampleToVol(current_k);
    net.forward(current_x, false);

    var last = net.layers[net.layers.length - 1];
    document.getElementById('inshape').textContent = '28 x 28 x 1 (Vol)';
    document.getElementById('outshape').textContent =
      last.out_sx + ' x ' + last.out_sy + ' x ' + last.out_depth +
      '  (' + last.layer_type + ')';
    if(typeof labels !== 'undefined') {
      document.getElementById('inlabel').textContent = 'MNIST label: ' + labels[current_k];
    }

    drawInput(document.getElementById('incanvas'), current_x);
    drawVol(document.getElementById('outcanvas'), last.out_act);

    setStatus('Forward pass OK: ' + net.layers.length + ' layers. Showing last layer output.');
  } catch(err) {
    setStatus('Error: ' + err.message, true);
  }
}

function drawInput(canvas, x) {
  var scale = 6;
  canvas.width = x.sx * scale;
  canvas.height = x.sy * scale;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for(var xx=0; xx<x.sx; xx++) {
    for(var yy=0; yy<x.sy; yy++) {
      var v = Math.floor((x.get(xx, yy, 0) + 0.5) * 255);
      if(v < 0) v = 0;
      if(v > 255) v = 255;
      ctx.fillStyle = 'rgb(' + v + ',' + v + ',' + v + ')';
      ctx.fillRect(xx*scale, yy*scale, scale, scale);
    }
  }
}

// Render a Vol as an auto-scaled grayscale heatmap, tiling its depth channels.
// Small spatial volumes (fc/softmax outputs) are drawn at a larger scale.
function drawVol(canvas, A) {
  var mm = convnetjs.maxmin(A.w);
  var dv = (mm.dv === 0) ? 1 : mm.dv;
  var scale = (A.sx <= 2 && A.sy <= 2) ? 20 : 4;
  var tileW = A.sx * scale;
  var tileH = A.sy * scale;
  var cols = Math.max(1, Math.min(A.depth, Math.floor(600 / tileW)));
  var rows = Math.ceil(A.depth / cols);
  canvas.width = cols * tileW;
  canvas.height = rows * tileH;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for(var d=0; d<A.depth; d++) {
    var ox = (d % cols) * tileW;
    var oy = Math.floor(d / cols) * tileH;
    for(var xx=0; xx<A.sx; xx++) {
      for(var yy=0; yy<A.sy; yy++) {
        var val = Math.floor((A.get(xx, yy, d) - mm.minv) / dv * 255);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        ctx.fillStyle = 'rgb(' + val + ',' + val + ',' + val + ')';
        ctx.fillRect(ox + xx*scale, oy + yy*scale, scale, scale);
      }
    }
  }
}

$(function() {
  document.getElementById('layerdef').value = default_layerdefs;
  loadData();
});
