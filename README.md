# GPU Direct Storage (GDS) Benchmark

A comprehensive benchmark suite for analyzing latency and throughput of GPU Direct Storage workloads compared to traditional I/O paths.

## Features

- **Latency Analysis**: Measures per-operation latency with percentile statistics (P50, P95, P99, P99.9)
- **Throughput Measurement**: Calculates bandwidth (GB/s) and IOPS
- **Multiple Block Sizes**: Tests from 4KB to 128MB to cover different workload patterns
- **Random I/O Patterns**: Simulates real-world random access patterns
- **Comparison**: Compares GDS vs traditional CPU-mediated I/O
- **CSV Export**: Outputs detailed results for further analysis
- **PCIe Utilization**: Shows percentage of theoretical PCIe bandwidth utilized

## Requirements

### Hardware
- NVIDIA GPU with GPUDirect Storage support (Ampere or newer recommended)
- NVMe SSD or other storage with GDS support
- PCIe Gen3 or Gen4 connection

### Software
- NVIDIA GPU Driver with GDS support (450.80.02 or later)
- CUDA Toolkit 11.4 or later
- GPUDirect Storage (cuFile library)
- Linux kernel with GDS support

### Installation

1. **Install NVIDIA GPU Driver**:
```bash
# Check current driver version
nvidia-smi

# Driver should be 450.80.02 or later
```

2. **Install CUDA Toolkit**:
```bash
# Download from https://developer.nvidia.com/cuda-downloads
# Or use package manager
sudo apt-get install cuda-toolkit-12-0
```

3. **Install GPUDirect Storage**:
```bash
# Download from https://developer.nvidia.com/gpudirect-storage
# Follow installation instructions
sudo apt-get install nvidia-gds
```

4. **Verify GDS Installation**:
```bash
# Check if cuFile library is available
ldconfig -p | grep cufile

# Check GDS status
/usr/local/cuda/gds/tools/gdscheck -p
```

## Building

### Basic Build
```bash
make benchmark
```

### Specify GPU Architecture
```bash
# For A100
make benchmark CUDA_ARCH=sm_80

# For H100
make benchmark CUDA_ARCH=sm_90

# For RTX 3090
make benchmark CUDA_ARCH=sm_86
```

### Build Options
```bash
make all        # Build both original and benchmark programs
make original   # Build original gds_test
make benchmark  # Build new benchmark program
make clean      # Remove build artifacts
make help       # Show help
```

## Usage

### Command-Line Options
```bash
./gds_benchmark [OPTIONS]

Options:
  -h, --help              Show help message
  -s, --size SIZE         File size in GB (default: 4)
  -f, --file PATH         Test file path (default: /mnt/tmp/gds_benchmark.dat)
  -b, --block SIZE        Block size in KB (default: test all sizes)
  -q, --queue DEPTH       Queue depth (default: 1)
  -i, --iterations NUM    Number of iterations per test (default: auto)
  -p, --pattern PATTERN   I/O pattern: random, sequential, or both (default: both)
```

### Examples

**Show help**:
```bash
./gds_benchmark --help
```

**Default (4GB file, all block sizes, both patterns)**:
```bash
./gds_benchmark
```

**Custom file size (8GB)**:
```bash
./gds_benchmark -s 8
```

**Test only sequential I/O**:
```bash
./gds_benchmark -p sequential
```

**Test only random I/O**:
```bash
./gds_benchmark -p random
```

**Test sequential I/O with specific block size**:
```bash
./gds_benchmark -p sequential -b 4096
```

**Custom file size and path**:
```bash
./gds_benchmark -s 8 -f /mnt/nvme0/benchmark.dat
```

**Test specific block size (4MB only, both patterns)**:
```bash
./gds_benchmark -b 4096
```

**Test 64KB blocks with 10000 iterations (random only)**:
```bash
./gds_benchmark -b 64 -i 10000 -p random
```

**Full custom configuration**:
```bash
./gds_benchmark -s 8 -f /nvme/test.dat -b 1024 -p sequential -i 5000
```

## Output

### Console Output

The benchmark outputs detailed statistics for each test:

```
=== GDS Random Read ===
Configuration:
  Block Size: 4 KB
  Queue Depth: 1
  Iterations: 10000
  Pattern: Random
  Operation: Read

Latency (microseconds):
  Min:         45.23 μs
  Avg:         52.17 μs
  P50:         51.80 μs
  P95:         58.32 μs
  P99:         62.45 μs
  P99.9:       78.92 μs
  Max:        125.67 μs

Throughput:
  Bandwidth:       0.75 GB/s
  IOPS:       19163.45 ops/s
  PCIe Util:       2.34 %
  Total Time:      0.522 s
  Total Data:      0.39 GB
```

### CSV Export

Results are automatically saved to `gds_benchmark_results.csv` with columns:
- Method (GDS or Traditional)
- Operation (Read or Write)
- Pattern (Random or Sequential)
- Block Size (KB)
- Queue Depth
- Iterations
- Latency metrics (Min, Avg, P50, P95, P99, P99.9, Max in microseconds)
- Throughput (GB/s)
- IOPS
- PCIe Bandwidth Utilization (%)
- Total Time (seconds)

## Benchmark Tests

The benchmark runs the following tests for each block size:

### Random I/O Tests (when `-p random` or `-p both`)
1. **GDS Random Read**: Direct GPU-to-Storage read with random access
2. **Traditional Random Read**: CPU-mediated read (Storage → CPU → GPU)
3. **GDS Random Write**: Direct GPU-to-Storage write with random access
4. **Traditional Random Write**: CPU-mediated write (GPU → CPU → Storage)

### Sequential I/O Tests (when `-p sequential` or `-p both`)
5. **GDS Sequential Read**: Direct GPU-to-Storage read with sequential access
6. **Traditional Sequential Read**: CPU-mediated sequential read
7. **GDS Sequential Write**: Direct GPU-to-Storage write with sequential access
8. **Traditional Sequential Write**: CPU-mediated sequential write

### I/O Pattern Differences
- **Random Access**: Reads/writes from random file offsets (simulates database/metadata workloads)
- **Sequential Access**: Reads/writes data sequentially from start (simulates large data transfers, ML model loading)

### Block Sizes Tested
- 4 KB
- 16 KB
- 64 KB
- 256 KB
- 1 MB
- 4 MB
- 16 MB
- 64 MB
- 128 MB

### Performance Expectations
- **Sequential I/O**: Typically achieves higher throughput, especially for large block sizes
- **Random I/O**: More latency-sensitive, benefits more from GDS at smaller block sizes
- **GDS Advantage**: Most pronounced for large transfers and when CPU is a bottleneck

## Performance Tips

### Storage Configuration

1. **Use O_DIRECT capable filesystem**:
```bash
# XFS is recommended
sudo mkfs.xfs /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme
```

2. **Verify GDS is working**:
```bash
# Check GDS configuration
cat /etc/cufile.json

# Verify storage is GDS-compatible
/usr/local/cuda/gds/tools/gdscheck -p
```

3. **Disable filesystem cache for testing**:
```bash
# Clear cache before each run
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### System Configuration

1. **Set CPU governor to performance**:
```bash
sudo cpupower frequency-set -g performance
```

2. **Disable GPU power management**:
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 400  # Set appropriate power limit
```

3. **Check PCIe link status**:
```bash
nvidia-smi -q | grep -A 3 "PCIe"
```

## Understanding Results

### Latency Metrics
- **Min/Max**: Best and worst case latency
- **Avg**: Mean latency across all operations
- **P50**: Median latency (50th percentile)
- **P95**: 95% of operations complete within this time
- **P99**: 99% of operations complete within this time
- **P99.9**: Tail latency (1 in 1000 operations)

### When to Use GDS
GDS shows the most benefit for:
- Large block sizes (>1MB)
- High throughput workloads
- GPU compute + storage intensive applications
- Scenarios where CPU is a bottleneck

Traditional I/O may be competitive for:
- Very small block sizes (<64KB)
- When filesystem cache helps
- Sequential access patterns
- When CPU is not saturated

### PCIe Bandwidth Utilization
- PCIe Gen3 x16: ~32 GB/s theoretical
- PCIe Gen4 x16: ~64 GB/s theoretical
- PCIe Gen5 x16: ~128 GB/s theoretical

Lower utilization for small block sizes is expected due to latency overhead.

## Troubleshooting

### "cuFile driver open failed"
- Verify GDS is installed: `dpkg -l | grep nvidia-gds`
- Check driver version: `nvidia-smi`
- Verify GDS configuration: `gdscheck -p`

### "Failed to open file with O_DIRECT"
- Ensure filesystem supports O_DIRECT (XFS, ext4)
- Check file path exists and has write permissions
- Verify block size alignment (must be 4KB aligned)

### Low Performance
- Check PCIe link speed: `nvidia-smi -q | grep PCIe`
- Verify storage device supports GDS: `gdscheck -p`
- Clear filesystem cache: `echo 3 > /proc/sys/vm/drop_caches`
- Check CPU governor: `cpupower frequency-info`

### Compilation Errors
- Verify CUDA toolkit is installed
- Check cuFile library path: `ldconfig -p | grep cufile`
- Update CUDA_ARCH in Makefile to match your GPU

## Advanced Usage

### Analyzing Results with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('gds_benchmark_results.csv')

# Plot latency comparison
gds_read = df[(df['Method'] == 'GDS') & (df['Operation'] == 'Read')]
trad_read = df[(df['Method'] == 'Traditional') & (df['Operation'] == 'Read')]

plt.figure(figsize=(12, 6))
plt.semilogx(gds_read['BlockSize_KB'], gds_read['P99_us'], 'o-', label='GDS')
plt.semilogx(trad_read['BlockSize_KB'], trad_read['P99_us'], 's-', label='Traditional')
plt.xlabel('Block Size (KB)')
plt.ylabel('P99 Latency (μs)')
plt.title('Latency Comparison: GDS vs Traditional I/O')
plt.legend()
plt.grid(True)
plt.show()
```

## References

- [NVIDIA GPUDirect Storage Documentation](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/)
- [GDS Best Practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/)

## License

This benchmark is provided as-is for testing and evaluation purposes.
