(function(lib) {
  "use strict";
  if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = lib; // in nodejs
  } else {
    // browser, web worker, or other host
    var g = (typeof globalThis !== "undefined") ? globalThis
          : (typeof window !== "undefined") ? window
          : this;
    g.convnetjs = lib;
  }
})(convnetjs);
