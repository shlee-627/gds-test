# Makefile for GDS Benchmark

# CUDA compiler
NVCC = nvcc

# CUDA architecture (adjust based on your GPU)
# Common values:
#   - sm_70 for V100
#   - sm_80 for A100
#   - sm_86 for RTX 3090
#   - sm_89 for RTX 4090
#   - sm_90 for H100
CUDA_ARCH = sm_80

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++11
NVCC_FLAGS_GDS = $(NVCC_FLAGS) -DUSE_CUFILE

# Libraries
LIBS =
LIBS_GDS = -lcufile

# Include paths (adjust if needed)
INCLUDE =
INCLUDE_GDS = -I/usr/local/cuda/targets/x86_64-linux/include

# Output binaries
TARGET_ORIGINAL = gds_test
TARGET_BENCHMARK = gds_benchmark

.PHONY: all clean original benchmark help

all: original benchmark

# Build original test program
original:
	$(NVCC) $(NVCC_FLAGS_GDS) $(INCLUDE_GDS) -o $(TARGET_ORIGINAL) gds_test.cu $(LIBS_GDS)

# Build new benchmark program
benchmark:
	$(NVCC) $(NVCC_FLAGS_GDS) $(INCLUDE_GDS) -o $(TARGET_BENCHMARK) gds_benchmark.cu $(LIBS_GDS)

# Build without GDS support (fallback)
no-gds:
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -o $(TARGET_BENCHMARK)_no_gds gds_benchmark.cu $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET_ORIGINAL) $(TARGET_BENCHMARK) $(TARGET_BENCHMARK)_no_gds
	rm -f *.o
	rm -f gds_benchmark_results.csv

# Help
help:
	@echo "GDS Benchmark Makefile"
	@echo "======================"
	@echo ""
	@echo "Targets:"
	@echo "  make all        - Build both original and benchmark programs (default)"
	@echo "  make original   - Build original gds_test program"
	@echo "  make benchmark  - Build new gds_benchmark program"
	@echo "  make no-gds     - Build benchmark without GDS support"
	@echo "  make clean      - Remove all build artifacts"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH      - Target GPU architecture (default: sm_80)"
	@echo "                   Set with: make CUDA_ARCH=sm_90 benchmark"
	@echo ""
	@echo "Common GPU architectures:"
	@echo "  sm_70  - V100"
	@echo "  sm_80  - A100"
	@echo "  sm_86  - RTX 3090"
	@echo "  sm_89  - RTX 4090"
	@echo "  sm_90  - H100"
	@echo ""
	@echo "Usage example:"
	@echo "  make benchmark CUDA_ARCH=sm_86"
	@echo "  ./gds_benchmark 4 /mnt/nvme/test.dat"
