NVCC      ?= nvcc
# -arch=native lets nvcc auto-detect whichever GPU is installed in the
# machine (works on Colab T4, on local Ampere/Ada/Hopper workstations,
# etc.). Override explicitly if needed, e.g. `make ARCH=sm_75`.
ARCH      ?= native
NVCCFLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr \
             -Xcompiler -march=native

all: a.out

a.out: main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f a.out

.PHONY: all clean
