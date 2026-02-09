#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#ifdef USE_CUFILE
#include <cufile.h>
#endif

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Configuration
struct BenchmarkConfig {
    size_t fileSize;
    size_t blockSize;
    int queueDepth;
    int numIterations;
    int runtimeSeconds;      // Time-based mode: run for N seconds (0 = use numIterations)
    const char* filename;
    bool isWrite;
    bool isRandom;
};

// Statistics structure
struct BenchmarkStats {
    std::vector<double> latencies;  // in microseconds
    double totalTime;                // in seconds
    size_t totalBytes;
    double avgLatency;
    double minLatency;
    double maxLatency;
    double p50Latency;
    double p95Latency;
    double p99Latency;
    double p999Latency;
    double throughput;               // GB/s
    double iops;
    double bandwidthUtil;            // percentage
};

// Get high-resolution time in seconds
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Calculate percentile from sorted vector
double calculatePercentile(std::vector<double>& sorted_data, double percentile) {
    if (sorted_data.empty()) return 0.0;

    size_t n = sorted_data.size();
    double index = (percentile / 100.0) * (n - 1);
    size_t lower = (size_t)floor(index);
    size_t upper = (size_t)ceil(index);

    if (lower == upper) {
        return sorted_data[lower];
    }

    double weight = index - lower;
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

// Calculate statistics from latency measurements
void calculateStats(BenchmarkStats& stats, const BenchmarkConfig& config,
                    double pcieBandwidthGBs = 32.0) {
    if (stats.latencies.empty()) return;

    // Sort latencies for percentile calculation
    std::vector<double> sorted = stats.latencies;
    std::sort(sorted.begin(), sorted.end());

    // Calculate min, max, avg
    stats.minLatency = sorted.front();
    stats.maxLatency = sorted.back();

    double sum = 0.0;
    for (double lat : stats.latencies) {
        sum += lat;
    }
    stats.avgLatency = sum / stats.latencies.size();

    // Calculate percentiles
    stats.p50Latency = calculatePercentile(sorted, 50.0);
    stats.p95Latency = calculatePercentile(sorted, 95.0);
    stats.p99Latency = calculatePercentile(sorted, 99.0);
    stats.p999Latency = calculatePercentile(sorted, 99.9);

    // Calculate throughput and IOPS
    stats.throughput = (stats.totalBytes / (1024.0*1024.0*1024.0)) / stats.totalTime;

		printf ("calculateStats: latencies sample size : %d\n", stats.latencies.size());
    stats.iops = stats.latencies.size() / stats.totalTime;

    // Calculate bandwidth utilization
    stats.bandwidthUtil = (stats.throughput / pcieBandwidthGBs) * 100.0;
}

// Print statistics
void printStats(const BenchmarkStats& stats, const BenchmarkConfig& config,
                const char* testName) {
    printf("\n=== %s ===\n", testName);
    printf("Configuration:\n");
    printf("  Block Size: %zu KB\n", config.blockSize / 1024);
    printf("  Queue Depth: %d\n", config.queueDepth);
    printf("  Iterations: %d\n", config.numIterations);
    printf("  Pattern: %s\n", config.isRandom ? "Random" : "Sequential");
    printf("  Operation: %s\n", config.isWrite ? "Write" : "Read");

    printf("\nLatency (microseconds):\n");
    printf("  Min:    %10.2f μs\n", stats.minLatency);
    printf("  Avg:    %10.2f μs\n", stats.avgLatency);
    printf("  P50:    %10.2f μs\n", stats.p50Latency);
    printf("  P95:    %10.2f μs\n", stats.p95Latency);
    printf("  P99:    %10.2f μs\n", stats.p99Latency);
    printf("  P99.9:  %10.2f μs\n", stats.p999Latency);
    printf("  Max:    %10.2f μs\n", stats.maxLatency);

    printf("\nThroughput:\n");
    printf("  Bandwidth:  %10.2f GB/s\n", stats.throughput);
    printf("  IOPS:       %10.2f ops/s\n", stats.iops);
    printf("  PCIe Util:  %10.2f %%\n", stats.bandwidthUtil);
    printf("  Total Time: %10.3f s\n", stats.totalTime);
    printf("  Total Data: %10.2f GB\n", stats.totalBytes / (1024.0*1024.0*1024.0));
}

// Create test file with specified size
void createTestFile(const char* filename, size_t size) {
    printf("Creating test file: %s (%.2f GB)\n",
           filename, size / (1024.0 * 1024.0 * 1024.0));

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to create file");
        exit(1);
    }

    // Allocate and fill buffer
    size_t chunkSize = 1024 * 1024;  // 1MB chunks
    char* buffer = (char*)malloc(chunkSize);
    memset(buffer, 0xAB, chunkSize);

    for (size_t written = 0; written < size; written += chunkSize) {
        size_t toWrite = (size - written < chunkSize) ? (size - written) : chunkSize;
        ssize_t ret = write(fd, buffer, toWrite);
        if (ret < 0) {
            perror("Failed to write file");
            free(buffer);
            close(fd);
            exit(1);
        }
    }

    free(buffer);
    close(fd);
    printf("File created successfully\n\n");
}

#ifdef USE_CUFILE
// GDS Benchmark - Single threaded with latency tracking
BenchmarkStats benchmarkGDS(const BenchmarkConfig& config) {
    BenchmarkStats stats = {};
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // Initialize cuFile
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return stats;
    }

    // Allocate GPU memory
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, config.blockSize * config.queueDepth));

    if (config.isWrite) {
        CHECK_CUDA(cudaMemset(d_buffer, 0xCD, config.blockSize * config.queueDepth));
    }

    // Open file
    int flags = config.isWrite ? (O_WRONLY | O_CREAT | O_DIRECT) : (O_RDONLY | O_DIRECT);
    int fd = open(config.filename, flags, 0644);
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT");
        cudaFree(d_buffer);
        cuFileDriverClose();
        return stats;
    }

    // Register cuFile handle
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        cudaFree(d_buffer);
        cuFileDriverClose();
        return stats;
    }

    // Prepare random offsets if needed
    std::vector<off_t> offsets;
    size_t maxOffset = (config.fileSize / config.blockSize) * config.blockSize;

    if (config.isRandom) {
        srand(12345);  // Fixed seed for reproducibility
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (rand() % (maxOffset / config.blockSize)) * config.blockSize;
            offsets.push_back(offset);
        }
    } else {
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (i * config.blockSize) % maxOffset;
            offsets.push_back(offset);
        }
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        if (config.isWrite) {
            cuFileWrite(cf_handle, d_buffer, config.blockSize, offsets[i % offsets.size()], 0);
        } else {
            cuFileRead(cf_handle, d_buffer, config.blockSize, offsets[i % offsets.size()], 0);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Actual benchmark
    double totalStart = getTime();

    for (int i = 0; i < config.numIterations; i++) {
        double opStart = getTime();

        ssize_t ret;
        if (config.isWrite) {
            ret = cuFileWrite(cf_handle, d_buffer, config.blockSize, offsets[i], 0);
        } else {
            ret = cuFileRead(cf_handle, d_buffer, config.blockSize, offsets[i], 0);
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        double opEnd = getTime();
        double latency_us = (opEnd - opStart) * 1e6;  // Convert to microseconds

        stats.latencies.push_back(latency_us);
        stats.totalBytes += config.blockSize;

        if (ret < 0) {
            fprintf(stderr, "cuFile operation failed at iteration %d\n", i);
            break;
        }
    }

    stats.totalTime = getTime() - totalStart;

    // Cleanup
    cuFileHandleDeregister(cf_handle);
    close(fd);
    CHECK_CUDA(cudaFree(d_buffer));
    cuFileDriverClose();

    return stats;
}

// GDS Benchmark - True Batch API (most efficient for QD > 1)
BenchmarkStats benchmarkGDSBatch(const BenchmarkConfig& config) {
    BenchmarkStats stats = {};
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // Initialize cuFile
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return stats;
    }

    // Allocate GPU memory for all in-flight operations
    char** d_buffers = new char*[config.queueDepth];
    for (int i = 0; i < config.queueDepth; i++) {
        CHECK_CUDA(cudaMalloc(&d_buffers[i], config.blockSize));
        if (config.isWrite) {
            CHECK_CUDA(cudaMemset(d_buffers[i], 0xCD, config.blockSize));
        }
    }

    // Open file
    int flags = config.isWrite ? (O_WRONLY | O_CREAT | O_DIRECT) : (O_RDONLY | O_DIRECT);
    int fd = open(config.filename, flags, 0644);
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT");
        for (int i = 0; i < config.queueDepth; i++) {
            cudaFree(d_buffers[i]);
        }
        delete[] d_buffers;
        cuFileDriverClose();
        return stats;
    }

    // Register cuFile handle
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        for (int i = 0; i < config.queueDepth; i++) {
            cudaFree(d_buffers[i]);
        }
        delete[] d_buffers;
        cuFileDriverClose();
        return stats;
    }

    // Prepare random offsets
    std::vector<off_t> offsets;
    size_t maxOffset = (config.fileSize / config.blockSize) * config.blockSize;

    if (config.isRandom) {
        srand(12345);
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (rand() % (maxOffset / config.blockSize)) * config.blockSize;
            offsets.push_back(offset);
        }
    } else {
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (i * config.blockSize) % maxOffset;
            offsets.push_back(offset);
        }
    }

    // Warmup - Skip several initial I/O requests (To avoid out-liers)
    for (int i = 0; i < 10; i++) {
        if (config.isWrite) {
            cuFileWrite(cf_handle, d_buffers[0], config.blockSize, offsets[i % offsets.size()], 0);
        } else {
            cuFileRead(cf_handle, d_buffers[0], config.blockSize, offsets[i % offsets.size()], 0);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Setup batch handle (REQUIRED for Batch API)
    CUfileBatchHandle_t batch_handle;
    status = cuFileBatchIOSetUp(&batch_handle, config.queueDepth);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileBatchIOSetUp failed: %d\n", status.err);
        cuFileHandleDeregister(cf_handle);
        close(fd);
        for (int i = 0; i < config.queueDepth; i++) {
            cudaFree(d_buffers[i]);
        }
        delete[] d_buffers;
        cuFileDriverClose();
        return stats;
    }

    // Setup batch I/O structures
    CUfileIOParams_t* io_batch = new CUfileIOParams_t[config.queueDepth];
    CUfileIOEvents_t* io_events = new CUfileIOEvents_t[config.queueDepth];

    // Initialize batch parameters
    for (int i = 0; i < config.queueDepth; i++) {
        memset(&io_batch[i], 0, sizeof(CUfileIOParams_t));
        io_batch[i].mode = CUFILE_BATCH;
        io_batch[i].fh = cf_handle;
        io_batch[i].u.batch.devPtr_base = d_buffers[i];
        io_batch[i].u.batch.devPtr_offset = 0;
        io_batch[i].u.batch.size = config.blockSize;
        io_batch[i].opcode = config.isWrite ? CUFILE_WRITE : CUFILE_READ;
        io_batch[i].cookie = (void*)(uintptr_t)i;  // Track which slot
    }

    // Tracking structures
    struct BatchOp {
        int slotId;
        double startTime;
        bool active;
    };

    std::vector<BatchOp> inFlightOps(config.queueDepth);
    for (int i = 0; i < config.queueDepth; i++) {
        inFlightOps[i].slotId = i;
        inFlightOps[i].active = false;
    }

    int nextOp = 0;
    int completedOps = 0;

    // Pre-allocate submit batch buffer (avoid new/delete in hot path)
    CUfileIOParams_t* submitBatch = new CUfileIOParams_t[config.queueDepth];

    double totalStart = getTime();
    bool timeBasedMode = (config.runtimeSeconds > 0);
    double benchmarkEndTime = totalStart + config.runtimeSeconds;

    // Main batch I/O loop - optimized for maximum IOPS
    // Time-based: run until time expires
    // Iteration-based: run until numIterations completed
    while (true) {
        // Check termination condition
        if (timeBasedMode) {
            if (getTime() >= endTime) {
                break;  // Time expired
            }
        } else {
            if (completedOps >= config.numIterations) {
                break;  // All iterations done
            }
        }

        // Prepare batch for submission
        int batchSize = 0;

        for (int i = 0; i < config.queueDepth; i++) {
            if (!inFlightOps[i].active) {
                int slotId = i;

                // Get next offset (wrap around for time-based mode)
                int offsetIndex = timeBasedMode ? (nextOp % offsets.size()) : nextOp;

                // Check if we have more iterations to submit (iteration-based only)
                if (!timeBasedMode && nextOp >= config.numIterations) {
                    break;
                }

                // Setup this batch entry
                io_batch[slotId].u.batch.file_offset = offsets[offsetIndex];

                // Record start time
                inFlightOps[slotId].startTime = getTime();
                inFlightOps[slotId].active = true;

                submitBatch[batchSize++] = io_batch[slotId];
                nextOp++;
            }
        }

        // Submit batch if we have operations
        if (batchSize > 0) {
            // Submit the batch (non-blocking) with correct API signature
            status = cuFileBatchIOSubmit(batch_handle, batchSize, submitBatch, 0);
            if (status.err != CU_FILE_SUCCESS) {
                fprintf(stderr, "Batch submit failed: %d\n", status.err);
                break;
            }
        }

        // Poll for completions with correct API signature
        unsigned int numEvents = config.queueDepth;
        status = cuFileBatchIOGetStatus(batch_handle, 0, &numEvents, io_events, NULL);

        if (status.err == CU_FILE_SUCCESS) {
            for (int i = 0; i < numEvents; i++) {
                if (io_events[i].status == CUFILE_COMPLETE) {
                    int slotId = (int)(uintptr_t)io_events[i].cookie;

                    if (inFlightOps[slotId].active) {
                        double endTime = getTime();
                        double latency_us = (endTime - inFlightOps[slotId].startTime) * 1e6;

                        stats.latencies.push_back(latency_us);
                        stats.totalBytes += config.blockSize;

                        inFlightOps[slotId].active = false;
                        completedOps++;
                    }
                }
            }
        }

        // Busy poll for maximum IOPS (no sleep)
        // Note: This uses 100% CPU. For lower CPU usage, add: usleep(1);
    }

    // Cleanup pre-allocated buffer
    delete[] submitBatch;

    // Wait for all remaining operations
    CHECK_CUDA(cudaDeviceSynchronize());

    stats.totalTime = getTime() - totalStart;

    // Cleanup
    delete[] io_batch;
    delete[] io_events;

    // Destroy batch handle
    cuFileBatchIODestroy(batch_handle);

    cuFileHandleDeregister(cf_handle);
    close(fd);

    for (int i = 0; i < config.queueDepth; i++) {
        CHECK_CUDA(cudaFree(d_buffers[i]));
    }
    delete[] d_buffers;

    cuFileDriverClose();

    return stats;
}

// GDS Benchmark - Asynchronous with batch API for queue depth > 1
BenchmarkStats benchmarkGDSAsync(const BenchmarkConfig& config) {
    BenchmarkStats stats = {};
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // Initialize cuFile
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return stats;
    }

    // Allocate GPU memory for all in-flight operations
    char** d_buffers = new char*[config.queueDepth];
    for (int i = 0; i < config.queueDepth; i++) {
        CHECK_CUDA(cudaMalloc(&d_buffers[i], config.blockSize));
        if (config.isWrite) {
            CHECK_CUDA(cudaMemset(d_buffers[i], 0xCD, config.blockSize));
        }
    }

    // Create CUDA streams for async operations
    cudaStream_t* streams = new cudaStream_t[config.queueDepth];
    for (int i = 0; i < config.queueDepth; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Open file
    int flags = config.isWrite ? (O_WRONLY | O_CREAT | O_DIRECT) : (O_RDONLY | O_DIRECT);
    int fd = open(config.filename, flags, 0644);
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT");
        for (int i = 0; i < config.queueDepth; i++) {
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }
        delete[] d_buffers;
        delete[] streams;
        cuFileDriverClose();
        return stats;
    }

    // Register cuFile handle
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        for (int i = 0; i < config.queueDepth; i++) {
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }
        delete[] d_buffers;
        delete[] streams;
        cuFileDriverClose();
        return stats;
    }

    // Prepare random offsets
    std::vector<off_t> offsets;
    size_t maxOffset = (config.fileSize / config.blockSize) * config.blockSize;

    if (config.isRandom) {
        srand(12345);
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (rand() % (maxOffset / config.blockSize)) * config.blockSize;
            offsets.push_back(offset);
        }
    } else {
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (i * config.blockSize) % maxOffset;
            offsets.push_back(offset);
        }
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        if (config.isWrite) {
            cuFileWrite(cf_handle, d_buffers[0], config.blockSize, offsets[i % offsets.size()], 0);
        } else {
            cuFileRead(cf_handle, d_buffers[0], config.blockSize, offsets[i % offsets.size()], 0);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Tracking structures for async operations
    struct AsyncOp {
        int slotId;
        double startTime;
        bool active;
    };

    std::vector<AsyncOp> inFlightOps(config.queueDepth);
    for (int i = 0; i < config.queueDepth; i++) {
        inFlightOps[i].slotId = i;
        inFlightOps[i].active = false;
    }

    int nextOp = 0;           // Next operation to submit
    int completedOps = 0;     // Number of completed operations

    double totalStart = getTime();

    // Main async I/O loop
    while (completedOps < config.numIterations) {
        // Submit new operations to fill queue depth
        for (int i = 0; i < config.queueDepth && nextOp < config.numIterations; i++) {
            if (!inFlightOps[i].active) {
                int slotId = i;

                // Record start time for this operation
                inFlightOps[slotId].startTime = getTime();
                inFlightOps[slotId].active = true;

                // Issue async I/O operation
                ssize_t ret;
                if (config.isWrite) {
                    ret = cuFileWrite(cf_handle, d_buffers[slotId], config.blockSize,
                                     offsets[nextOp], 0);
                } else {
                    ret = cuFileRead(cf_handle, d_buffers[slotId], config.blockSize,
                                    offsets[nextOp], 0);
                }

                if (ret < 0) {
                    fprintf(stderr, "cuFile operation failed at iteration %d\n", nextOp);
                    inFlightOps[slotId].active = false;
                    break;
                }

                nextOp++;
            }
        }

        // Check for completions (poll)
        for (int i = 0; i < config.queueDepth; i++) {
            if (inFlightOps[i].active) {
                // Query stream status
                cudaError_t streamStatus = cudaStreamQuery(streams[i]);

                if (streamStatus == cudaSuccess) {
                    // Operation completed
                    double endTime = getTime();
                    double latency_us = (endTime - inFlightOps[i].startTime) * 1e6;

                    stats.latencies.push_back(latency_us);
                    stats.totalBytes += config.blockSize;

                    inFlightOps[i].active = false;
                    completedOps++;
                } else if (streamStatus != cudaErrorNotReady) {
                    // Error occurred
                    fprintf(stderr, "CUDA stream error: %s\n", cudaGetErrorString(streamStatus));
                    inFlightOps[i].active = false;
                    completedOps++;
                }
            }
        }

        // Small sleep to avoid busy-waiting
        usleep(1);
    }

    // Wait for all remaining operations to complete
    CHECK_CUDA(cudaDeviceSynchronize());

    stats.totalTime = getTime() - totalStart;

    // Cleanup
    cuFileHandleDeregister(cf_handle);
    close(fd);

    for (int i = 0; i < config.queueDepth; i++) {
        CHECK_CUDA(cudaFree(d_buffers[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    delete[] d_buffers;
    delete[] streams;

    cuFileDriverClose();

    return stats;
}

// Traditional I/O benchmark for comparison
BenchmarkStats benchmarkTraditional(const BenchmarkConfig& config) {
    BenchmarkStats stats = {};

    // Allocate host memory
    char* h_buffer = (char*)malloc(config.blockSize);
    if (!h_buffer) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return stats;
    }

    // Allocate GPU memory
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, config.blockSize));

    if (config.isWrite) {
        CHECK_CUDA(cudaMemset(d_buffer, 0xCD, config.blockSize));
    }

    // Open file
    int flags = config.isWrite ? (O_WRONLY | O_CREAT) : O_RDONLY;
    int fd = open(config.filename, flags, 0644);
    if (fd < 0) {
        perror("Failed to open file");
        free(h_buffer);
        cudaFree(d_buffer);
        return stats;
    }

    // Prepare random offsets
    std::vector<off_t> offsets;
    size_t maxOffset = (config.fileSize / config.blockSize) * config.blockSize;

    if (config.isRandom) {
        srand(12345);
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (rand() % (maxOffset / config.blockSize)) * config.blockSize;
            offsets.push_back(offset);
        }
    } else {
        for (int i = 0; i < config.numIterations; i++) {
            off_t offset = (i * config.blockSize) % maxOffset;
            offsets.push_back(offset);
        }
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        lseek(fd, offsets[i % offsets.size()], SEEK_SET);
        if (config.isWrite) {
            CHECK_CUDA(cudaMemcpy(h_buffer, d_buffer, config.blockSize, cudaMemcpyDeviceToHost));
            ssize_t ret = write(fd, h_buffer, config.blockSize);
            (void)ret;  // Suppress warning - warmup phase
        } else {
            ssize_t ret = read(fd, h_buffer, config.blockSize);
            (void)ret;  // Suppress warning - warmup phase
            CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, config.blockSize, cudaMemcpyHostToDevice));
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Actual benchmark
    double totalStart = getTime();

    for (int i = 0; i < config.numIterations; i++) {
        double opStart = getTime();

        lseek(fd, offsets[i], SEEK_SET);

        ssize_t ret;
        if (config.isWrite) {
            CHECK_CUDA(cudaMemcpy(h_buffer, d_buffer, config.blockSize, cudaMemcpyDeviceToHost));
            ret = write(fd, h_buffer, config.blockSize);
        } else {
            ret = read(fd, h_buffer, config.blockSize);
            CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, config.blockSize, cudaMemcpyHostToDevice));
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        double opEnd = getTime();
        double latency_us = (opEnd - opStart) * 1e6;

        stats.latencies.push_back(latency_us);
        stats.totalBytes += config.blockSize;

        if (ret < 0) {
            fprintf(stderr, "I/O error at iteration %d\n", i);
            break;
        }
    }

    stats.totalTime = getTime() - totalStart;

    // Cleanup
    close(fd);
    free(h_buffer);
    CHECK_CUDA(cudaFree(d_buffer));

    return stats;
}
#endif

// Print help message
void printHelp(const char* programName) {
    printf("GPU Direct Storage Latency & Throughput Benchmark\n");
    printf("==================================================\n\n");
    printf("Usage: %s [OPTIONS]\n\n", programName);
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -s, --size SIZE         File size in GB (default: 4)\n");
    printf("  -f, --file PATH         Test file path (default: /mnt/tmp/gds_benchmark.dat)\n");
    printf("  -b, --block SIZE        Block size in KB (default: test all sizes)\n");
    printf("                          Examples: 4, 64, 1024, 4096\n");
    printf("  -q, --queue DEPTH       Queue depth for async I/O (default: 1)\n");
    printf("                          QD=1: Synchronous I/O, QD>1: Asynchronous I/O\n");
    printf("                          Recommended: 4, 8, 16, 32 for high throughput\n");
    printf("  -i, --iterations NUM    Number of iterations per test (default: auto)\n");
    printf("  -t, --time SECONDS      Time-based mode: run for N seconds (overrides -i)\n");
    printf("                          Example: -t 30 (run for 30 seconds)\n");
    printf("  -p, --pattern PATTERN   I/O pattern: random, sequential, or both (default: both)\n");
    printf("                          Examples: random, sequential, both\n");
    printf("\n");
    printf("Block Size Presets:\n");
    printf("  If -b is not specified, tests all sizes: 4KB, 16KB, 64KB, 256KB,\n");
    printf("  1MB, 4MB, 16MB, 64MB, 128MB\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                           # Run with defaults (both patterns)\n", programName);
    printf("  %s -s 8                      # 8GB test file\n", programName);
    printf("  %s -s 4 -f /nvme/test.dat    # Custom file path\n", programName);
    printf("  %s -b 4096                   # Test only 4MB block size\n", programName);
    printf("  %s -b 64 -i 10000            # 64KB blocks, 10000 iterations\n", programName);
    printf("  %s -t 30                     # Run for 30 seconds (time-based)\n", programName);
    printf("  %s -b 4 -q 128 -t 60         # 4KB blocks, QD=128, run 60 seconds\n", programName);
    printf("  %s -p sequential             # Test only sequential I/O\n", programName);
    printf("  %s -p random -b 4            # Test only random I/O with 4KB blocks\n", programName);
    printf("  %s -q 16                     # Test with async I/O, queue depth 16\n", programName);
    printf("  %s -s 8 -b 1024 -q 32        # 8GB file, 1MB blocks, QD=32 (async)\n", programName);
    printf("\n");
    printf("Output:\n");
    printf("  - Console: Detailed statistics for each test\n");
    printf("  - CSV: gds_benchmark_results.csv (all metrics for analysis)\n");
    printf("\n");
}

// Parse size with unit support (e.g., "4096" or "4M" or "1G")
size_t parseSize(const char* str) {
    char* endptr;
    long long value = strtoll(str, &endptr, 10);

    if (endptr == str) {
        fprintf(stderr, "Invalid size: %s\n", str);
        exit(1);
    }

    // Check for unit suffix (K, M, G)
    if (*endptr == 'K' || *endptr == 'k') {
        value *= 1024;
    } else if (*endptr == 'M' || *endptr == 'm') {
        value *= 1024 * 1024;
    } else if (*endptr == 'G' || *endptr == 'g') {
        value *= 1024 * 1024 * 1024;
    }

    return (size_t)value;
}

int main(int argc, char** argv) {
    // Default parameters
    size_t fileSize = 4ULL * 1024 * 1024 * 1024;  // 4GB
    const char* filename = "/mnt/tmp/gds_benchmark.dat";
    int customBlockSize = -1;  // -1 means test all sizes
    int queueDepth = 1;
    int customIterations = -1;  // -1 means auto-calculate
    int runtimeSeconds = 0;      // 0 means iteration-based, >0 means time-based
    const char* pattern = "both";  // "random", "sequential", or "both"

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                fileSize = atoll(argv[++i]) * 1024ULL * 1024 * 1024;
            } else {
                fprintf(stderr, "Error: -s/--size requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                filename = argv[++i];
            } else {
                fprintf(stderr, "Error: -f/--file requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block") == 0) {
            if (i + 1 < argc) {
                customBlockSize = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: -b/--block requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--queue") == 0) {
            if (i + 1 < argc) {
                queueDepth = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: -q/--queue requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) {
                customIterations = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: -i/--iterations requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--time") == 0) {
            if (i + 1 < argc) {
                runtimeSeconds = atoi(argv[++i]);
                if (runtimeSeconds <= 0) {
                    fprintf(stderr, "Error: runtime must be > 0\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: -t/--time requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pattern") == 0) {
            if (i + 1 < argc) {
                pattern = argv[++i];
                if (strcmp(pattern, "random") != 0 &&
                    strcmp(pattern, "sequential") != 0 &&
                    strcmp(pattern, "both") != 0) {
                    fprintf(stderr, "Error: pattern must be 'random', 'sequential', or 'both'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: -p/--pattern requires an argument\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Use -h or --help for usage information\n");
            return 1;
        }
    }

    printf("GPU Direct Storage Latency & Throughput Benchmark\n");
    printf("==================================================\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Estimate PCIe bandwidth (GB/s)
    double pcieBandwidth = 32.0;  // Default PCIe Gen3 x16
    if (prop.major >= 8) {
        pcieBandwidth = 64.0;  // Assume PCIe Gen4 for newer GPUs
    }
    printf("Estimated PCIe Bandwidth: %.1f GB/s\n\n", pcieBandwidth);

#ifdef USE_CUFILE
    // Print configuration
    printf("Configuration:\n");
    printf("  Test File: %s\n", filename);
    printf("  File Size: %.2f GB\n", fileSize / (1024.0*1024.0*1024.0));
    if (customBlockSize > 0) {
        printf("  Block Size: %d KB\n", customBlockSize);
    } else {
        printf("  Block Sizes: 4KB to 128MB (9 sizes)\n");
    }
    printf("  Queue Depth: %d\n", queueDepth);
    printf("  I/O Pattern: %s\n", pattern);
    if (runtimeSeconds > 0) {
        printf("  Mode: Time-based (%d seconds)\n", runtimeSeconds);
    } else if (customIterations > 0) {
        printf("  Mode: Iteration-based (%d iterations)\n", customIterations);
    } else {
        printf("  Mode: Iteration-based (auto: min 100, max 10000)\n");
    }
    printf("\n");

    // Determine which patterns to test
    bool testRandom = (strcmp(pattern, "random") == 0 || strcmp(pattern, "both") == 0);
    bool testSequential = (strcmp(pattern, "sequential") == 0 || strcmp(pattern, "both") == 0);

    // Create test file
    createTestFile(filename, fileSize);

    // Block sizes to test
    std::vector<size_t> blockSizes;
    if (customBlockSize > 0) {
        // User specified a single block size (in KB)
        blockSizes.push_back(customBlockSize * 1024ULL);
    } else {
        // Test all default sizes (4KB to 128MB)
        blockSizes = {
            4 * 1024,           // 4KB
            16 * 1024,          // 16KB
            64 * 1024,          // 64KB
            256 * 1024,         // 256KB
            1 * 1024 * 1024,    // 1MB
            4 * 1024 * 1024,    // 4MB
            16 * 1024 * 1024,   // 16MB
            64 * 1024 * 1024,   // 64MB
            128 * 1024 * 1024   // 128MB
        };
    }

    // CSV output file
    FILE* csvFile = fopen("gds_benchmark_results.csv", "w");
    if (csvFile) {
        fprintf(csvFile, "Method,Operation,Pattern,BlockSize_KB,QueueDepth,Iterations,");
        fprintf(csvFile, "MinLatency_us,AvgLatency_us,P50_us,P95_us,P99_us,P999_us,MaxLatency_us,");
        fprintf(csvFile, "Throughput_GBps,IOPS,BandwidthUtil_pct,TotalTime_s\n");
    }

    // Run benchmarks for different block sizes
    for (size_t blockSize : blockSizes) {
        // Calculate iterations to read at least 1GB of data (ignored in time-based mode)
        int iterations;
        if (runtimeSeconds > 0) {
            // Time-based mode: use large number for offset array generation
            iterations = 1000000;  // Generate 1M offsets to cycle through
        } else if (customIterations > 0) {
            iterations = customIterations;
        } else {
            iterations = std::max(100, (int)(1024*1024*1024 / blockSize));
            iterations = std::min(iterations, 10000);  // Cap at 10000
        }

        printf("\n" "========================================\n");
        printf("Block Size: %zu KB\n", blockSize / 1024);
        if (runtimeSeconds > 0) {
            printf("Mode: Time-based (%d seconds)\n", runtimeSeconds);
        } else {
            printf("Mode: Iteration-based (%d iterations)\n", iterations);
        }
        printf("========================================\n");

        // Random I/O Tests
        if (testRandom) {
            // Test 1: GDS Random Read
            {
                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, filename, false, true
                };
                // Use batch API if queue depth > 1
                BenchmarkStats stats;
                if (queueDepth > 1) {
                    stats = benchmarkGDSBatch(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Random Read (Batch API)");
                } else {
                    stats = benchmarkGDS(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Random Read");
                }

                if (csvFile) {
                    fprintf(csvFile, "GDS,Read,Random,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }
            }

            // Test 2: Traditional Random Read
            {
                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, filename, false, true
                };
                BenchmarkStats stats = benchmarkTraditional(config);
                calculateStats(stats, config, pcieBandwidth);
                printStats(stats, config, "Traditional Random Read");

                if (csvFile) {
                    fprintf(csvFile, "Traditional,Read,Random,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }
            }

            // Test 3: GDS Random Write
            {
                const char* writeFile = "/mnt/tmp/gds_benchmark_write_random.dat";
                createTestFile(writeFile, fileSize);

                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, writeFile, true, true
                };
                // Use batch API if queue depth > 1
                BenchmarkStats stats;
                if (queueDepth > 1) {
                    stats = benchmarkGDSBatch(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Random Write (Batch API)");
                } else {
                    stats = benchmarkGDS(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Random Write");
                }

                if (csvFile) {
                    fprintf(csvFile, "GDS,Write,Random,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }

                unlink(writeFile);
            }

            // Test 4: Traditional Random Write
            {
                const char* writeFile = "/mnt/tmp/gds_benchmark_write_trad_random.dat";
                createTestFile(writeFile, fileSize);

                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, writeFile, true, true
                };
                BenchmarkStats stats = benchmarkTraditional(config);
                calculateStats(stats, config, pcieBandwidth);
                printStats(stats, config, "Traditional Random Write");

                if (csvFile) {
                    fprintf(csvFile, "Traditional,Write,Random,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }

                unlink(writeFile);
            }
        }

        // Sequential I/O Tests
        if (testSequential) {
            // Test 5: GDS Sequential Read
            {
                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, filename, false, false
                };
                // Use batch API if queue depth > 1
                BenchmarkStats stats;
                if (queueDepth > 1) {
                    stats = benchmarkGDSBatch(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Sequential Read (Batch API)");
                } else {
                    stats = benchmarkGDS(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Sequential Read");
                }

                if (csvFile) {
                    fprintf(csvFile, "GDS,Read,Sequential,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }
            }

            // Test 6: Traditional Sequential Read
            {
                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, filename, false, false
                };
                BenchmarkStats stats = benchmarkTraditional(config);
                calculateStats(stats, config, pcieBandwidth);
                printStats(stats, config, "Traditional Sequential Read");

                if (csvFile) {
                    fprintf(csvFile, "Traditional,Read,Sequential,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }
            }

            // Test 7: GDS Sequential Write
            {
                const char* writeFile = "/mnt/tmp/gds_benchmark_write_seq.dat";
                createTestFile(writeFile, fileSize);

                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, writeFile, true, false
                };
                // Use batch API if queue depth > 1
                BenchmarkStats stats;
                if (queueDepth > 1) {
                    stats = benchmarkGDSBatch(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Sequential Write (Batch API)");
                } else {
                    stats = benchmarkGDS(config);
                    calculateStats(stats, config, pcieBandwidth);
                    printStats(stats, config, "GDS Sequential Write");
                }

                if (csvFile) {
                    fprintf(csvFile, "GDS,Write,Sequential,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }

                unlink(writeFile);
            }

            // Test 8: Traditional Sequential Write
            {
                const char* writeFile = "/mnt/tmp/gds_benchmark_write_trad_seq.dat";
                createTestFile(writeFile, fileSize);

                BenchmarkConfig config = {
                    fileSize, blockSize, queueDepth, iterations, runtimeSeconds, writeFile, true, false
                };
                BenchmarkStats stats = benchmarkTraditional(config);
                calculateStats(stats, config, pcieBandwidth);
                printStats(stats, config, "Traditional Sequential Write");

                if (csvFile) {
                    fprintf(csvFile, "Traditional,Write,Sequential,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
                            blockSize/1024, queueDepth, iterations, stats.minLatency, stats.avgLatency,
                            stats.p50Latency, stats.p95Latency, stats.p99Latency, stats.p999Latency,
                            stats.maxLatency, stats.throughput, stats.iops, stats.bandwidthUtil, stats.totalTime);
                }

                unlink(writeFile);
            }
        }
    }

    if (csvFile) {
        fclose(csvFile);
        printf("\n\nResults saved to: gds_benchmark_results.csv\n");
    }

    // Cleanup
    unlink(filename);

    printf("\n" "Benchmark completed!\n");

#else
    printf("ERROR: This benchmark requires GPUDirect Storage support.\n");
    printf("Please compile with: -DUSE_CUFILE -lcufile\n");
    return 1;
#endif

    return 0;
}
