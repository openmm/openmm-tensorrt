.PHONY: standard clean check conda docker

# Set the plugin version from the git tags
export PLUGIN_VERSION=$(shell git describe --tags --dirty)

PLUGIN_CUDA_VERSION ?= 10.0
PLUGIN_DEVICE ?= 0

PLUGIN_CUDA_LABEL = cuda$(shell echo $(PLUGIN_CUDA_VERSION) | sed 's/\.//g')
PLUGIN_DOCKER_IMAGE = openmm-tensorrt-$(PLUGIN_CUDA_LABEL)

BUILD_DIR=build

standard:
	mkdir -p $(BUILD_DIR) &&\
	cd $(BUILD_DIR) &&\
	cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DEXTRA_COMPILE_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
		-DOPENMM_DIR=$(CONDA_PREFIX) \
		&&\
	make -j

clean:
	make -C $(BUILD_DIR) clean

check:
	make -C $(BUILD_DIR) test

docker_image:
	nvidia-docker build --tag $(PLUGIN_DOCKER_IMAGE) \
	                    --build-arg UID=$(shell id --user) \
	                    --build-arg GID=$(shell id --group) \
	                    --build-arg PLUGIN_CUDA_VERSION=$(PLUGIN_CUDA_VERSION) \
	                    --build-arg PLUGIN_CUDA_LABEL=$(PLUGIN_CUDA_LABEL) \
	                    docker

docker: docker_image
	NV_GPU=$(PLUGIN_DEVICE) \
	nvidia-docker run --tty --interactive --rm \
	                  --volume $(shell pwd):/home/user/openmm-tensorrt.git \
	                  $(PLUGIN_DOCKER_IMAGE)