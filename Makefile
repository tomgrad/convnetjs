# Build convnet.js by concatenating src/ in dependency order, then minify with esbuild.
# Requires only node/npx. `make clean` removes both generated outputs.

# Order matters: later files reference globals defined by earlier ones.
# convnet_init.js must be first; convnet_export.js must be last. Add new files here.
SRCS := \
	src/convnet_init.js \
	src/convnet_util.js \
	src/convnet_vol.js \
	src/convnet_vol_util.js \
	src/convnet_layers_dotproducts.js \
	src/convnet_layers_pool.js \
	src/convnet_layers_input.js \
	src/convnet_layers_loss.js \
	src/convnet_layers_nonlinearities.js \
	src/convnet_layers_dropout.js \
	src/convnet_layers_normalization.js \
	src/convnet_net.js \
	src/convnet_trainers.js \
	src/convnet_magicnet.js \
	src/convnet_export.js

.PHONY: all clean

all: build/convnet.js build/convnet-min.js

build/convnet.js: $(SRCS) Makefile
	@mkdir -p build
	awk 1 $(SRCS) > $@
	@echo "built $@"

build/convnet-min.js: build/convnet.js
	npx --yes esbuild $< --minify --outfile=$@
	@echo "built $@"

clean:
	rm -f build/convnet.js build/convnet-min.js
