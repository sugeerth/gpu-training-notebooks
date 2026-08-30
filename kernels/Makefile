# kernels/Makefile — build and run the CUDA kernels, with or without a GPU.
#
#   make            build every kernel (nvcc if you have it, the CPU shim if you don't)
#   make run        build and run every kernel, printing its table
#   make check      force the CPU build and run it — correctness only, no GPU, exit 1 on error
#   make 02_reduce  build and run one kernel
#   make clean
#
# On Colab (or any box with CUDA) `make run` is the whole story: nvcc is on PATH, -arch=native
# picks the right target for whatever card you were assigned, and each program prints its own
# measurements against that card's own peak numbers.

CXX      ?= g++
NVCC     ?= nvcc
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes)

# -arch=native needs CUDA >= 11.5. Override for older toolkits: make ARCH=sm_75
ARCH     ?= native
NVFLAGS  := -std=c++17 -O3 -I. -I../kernelbench/shim -arch=$(ARCH) -lineinfo
# -x c++ is what makes a .cu file ordinary C++; cuda_shim.hpp supplies the rest.
CXXFLAGS := -std=c++17 -O2 -I. -I../kernelbench/shim -pthread -Wno-unknown-pragmas -x c++

SRCS  := $(sort $(wildcard [0-9][0-9]_*.cu))
NAMES := $(basename $(SRCS))

ifeq ($(HAVE_NVCC),yes)
  BUILD := build/gpu
  MODE  := nvcc -arch=$(ARCH)
else
  BUILD := build/cpu
  MODE  := g++ via cuda_shim.hpp (correctness only, no timings)
endif

BINS := $(addprefix $(BUILD)/,$(NAMES))

.PHONY: all run check clean list $(NAMES)

all: $(BINS)
	@echo "built $(words $(BINS)) kernels with $(MODE)"

$(BUILD):
	@mkdir -p $@

SHIM := ../kernelbench/shim/common.cuh ../kernelbench/shim/cuda_shim.hpp

build/gpu/%: %.cu $(SHIM) | build/gpu
	$(NVCC) $(NVFLAGS) $< -o $@

build/cpu/%: %.cu $(SHIM) | build/cpu
	$(CXX) $(CXXFLAGS) $< -o $@

build/cpu:
	@mkdir -p $@

# `make 03_sgemm` — build one and run it.
$(NAMES): %: $(BUILD)/%
	@echo "=== $@ ==="
	@$(BUILD)/$@

run: $(BINS)
	@echo "mode: $(MODE)"
	@fail=0; for b in $(BINS); do \
	  echo ""; echo "=== $$(basename $$b) ==="; \
	  $$b || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo ""; echo "one or more kernels FAILED"; exit 1; fi; \
	echo ""; echo "all kernels passed"

# The CI gate. Always the CPU build, so it runs anywhere and so the shim's real barriers get
# to catch the race conditions that a GPU only exposes intermittently.
check:
	@$(MAKE) --no-print-directory HAVE_NVCC=no run

list:
	@echo $(NAMES) | tr ' ' '\n'

clean:
	@rm -rf build
