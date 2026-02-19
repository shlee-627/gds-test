#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
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

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void createTestFile(const char* filename, size_t size) {
    printf("Creating test file: %s (%.2f GB)\n", 
           filename, size / (1024.0 * 1024.0 * 1024.0));
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to create file");
        exit(1);
    }
    
    size_t chunkSize = 1024 * 1024;
    char* buffer = (char*)malloc(chunkSize);
    memset(buffer, 0xAB, chunkSize);
    
    for (size_t written = 0; written < size; written += chunkSize) {
        size_t toWrite = (size - written < chunkSize) ? (size - written) : chunkSize;
        fwrite(buffer, 1, toWrite, fp);
    }
    
    free(buffer);
    fclose(fp);
    printf("File created successfully\n\n");
}

#ifdef USE_CUFILE
// GDS Write: 4KB block size, QD=1 (sequential, one I/O at a time)
double benchmarkGDSWrite(const char* filename, size_t fileSize) {
    const size_t blockSize = 4096;  // 4KB
    const size_t numBlocks = fileSize / blockSize;

    printf("=== GPU Direct Storage Write Benchmark (4KB blocks, QD=1) ===\n");
    printf("  File size : %.2f GB\n", fileSize / (1024.0 * 1024.0 * 1024.0));
    printf("  Block size: %zu B\n",   blockSize);
    printf("  Num blocks: %zu\n\n",   numBlocks);

    // CPU-side wall-clock timers (for non-stream operations)
    double t0, t1;
    double time_driverOpen, time_fileOpen, time_handleRegister,
           time_handleDeregister, time_fileClose, time_driverClose;

    // GPU-side event timers (for CUDA stream operations)
    cudaEvent_t ev_malloc_start, ev_malloc_end;
    cudaEvent_t ev_memset_start, ev_memset_end;
    cudaEvent_t ev_sync_start,   ev_sync_end;
    cudaEvent_t ev_free_start,   ev_free_end;
    float gpu_ms_malloc, gpu_ms_memset, gpu_ms_sync, gpu_ms_free;

    CHECK_CUDA(cudaEventCreate(&ev_malloc_start));
    CHECK_CUDA(cudaEventCreate(&ev_malloc_end));
    CHECK_CUDA(cudaEventCreate(&ev_memset_start));
    CHECK_CUDA(cudaEventCreate(&ev_memset_end));
    CHECK_CUDA(cudaEventCreate(&ev_sync_start));
    CHECK_CUDA(cudaEventCreate(&ev_sync_end));
    CHECK_CUDA(cudaEventCreate(&ev_free_start));
    CHECK_CUDA(cudaEventCreate(&ev_free_end));

    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // [Step 1] Initialize cuFile driver  (CPU-side: pure ioctl, no GPU stream)
    t0 = getTime();
    status = cuFileDriverOpen();
    t1 = getTime();
    time_driverOpen = t1 - t0;
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return -1;
    }

    // [Step 2] Allocate GPU memory (one 4KB buffer, reused for every block)
    //          GPU-side event: cudaMalloc touches GPU MMU
    char* d_buffer;
    CHECK_CUDA(cudaEventRecord(ev_malloc_start, 0));
    CHECK_CUDA(cudaMalloc(&d_buffer, blockSize));
    CHECK_CUDA(cudaEventRecord(ev_malloc_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_malloc_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_malloc, ev_malloc_start, ev_malloc_end));

    // [Step 3] Fill GPU buffer with dummy data  (GPU-side event: fill kernel)
    //          Same 4KB pattern is written for every block.
    CHECK_CUDA(cudaEventRecord(ev_memset_start, 0));
    CHECK_CUDA(cudaMemset(d_buffer, 0xCD, blockSize));
    CHECK_CUDA(cudaEventRecord(ev_memset_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_memset_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_memset, ev_memset_start, ev_memset_end));

    // [Step 4] Open file descriptor with O_DIRECT  (CPU-side: VFS/kernel call)
    t0 = getTime();
    int fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    t1 = getTime();
    time_fileOpen = t1 - t0;
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT for writing");
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }

    // [Step 5] Register cuFile handle  (CPU-side: ioctl into nvidia-fs)
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    t0 = getTime();
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    t1 = getTime();
    time_handleRegister = t1 - t0;
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }

    // [Step 6] Issue 4KB writes sequentially at QD=1
    //          Each cuFileWrite is a synchronous blocking ioctl containing
    //          the full PCIe DMA + NVMe write for that 4KB block.
    size_t totalBytesWritten = 0;
    double minLatency_us = 1e18, maxLatency_us = 0.0, sumLatency_us = 0.0;

    double totalStart = getTime();

    for (size_t i = 0; i < numBlocks; i++) {
        off_t fileOffset = (off_t)(i * blockSize);

        double blockStart = getTime();
        ssize_t ret = cuFileWrite(cf_handle, d_buffer, blockSize, fileOffset, 0);
        double blockEnd = getTime();

        if (ret < 0) {
            fprintf(stderr, "cuFileWrite failed at block %zu (offset %zu)\n", i, (size_t)fileOffset);
            break;
        }

        totalBytesWritten += (size_t)ret;

        double lat_us = (blockEnd - blockStart) * 1e6;
        sumLatency_us += lat_us;
        if (lat_us < minLatency_us) minLatency_us = lat_us;
        if (lat_us > maxLatency_us) maxLatency_us = lat_us;
    }

    // [Step 7] cudaDeviceSynchronize  (GPU-side event: residual GPU work
    //          after all cuFileWrites; expected ~0 ms for synchronous GDS)
    CHECK_CUDA(cudaEventRecord(ev_sync_start, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_sync_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_sync_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_sync, ev_sync_start, ev_sync_end));

    double totalTime = getTime() - totalStart;

    // [Step 8] Deregister cuFile handle  (CPU-side: ioctl into nvidia-fs)
    t0 = getTime();
    cuFileHandleDeregister(cf_handle);
    t1 = getTime();
    time_handleDeregister = t1 - t0;

    // [Step 9] Close file descriptor  (CPU-side: VFS)
    t0 = getTime();
    close(fd);
    t1 = getTime();
    time_fileClose = t1 - t0;

    // [Step 10] Free GPU memory  (GPU-side event: cudaFree flushes GPU MMU)
    CHECK_CUDA(cudaEventRecord(ev_free_start, 0));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaEventRecord(ev_free_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_free_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_free, ev_free_start, ev_free_end));

    // [Step 11] Close cuFile driver  (CPU-side: ioctl)
    t0 = getTime();
    cuFileDriverClose();
    t1 = getTime();
    time_driverClose = t1 - t0;

    // Destroy events
    CHECK_CUDA(cudaEventDestroy(ev_malloc_start));
    CHECK_CUDA(cudaEventDestroy(ev_malloc_end));
    CHECK_CUDA(cudaEventDestroy(ev_memset_start));
    CHECK_CUDA(cudaEventDestroy(ev_memset_end));
    CHECK_CUDA(cudaEventDestroy(ev_sync_start));
    CHECK_CUDA(cudaEventDestroy(ev_sync_end));
    CHECK_CUDA(cudaEventDestroy(ev_free_start));
    CHECK_CUDA(cudaEventDestroy(ev_free_end));

    // --- Report ---
    double gbSize    = (double)totalBytesWritten / (1024.0 * 1024.0 * 1024.0);
    double avgLat_us = sumLatency_us / (double)numBlocks;
    double iops      = (double)numBlocks / totalTime;

    printf("[GDS Write] Setup/Teardown Timing:\n");
    printf("  Timer key: [CPU] = wall-clock (CLOCK_MONOTONIC), [GPU] = cudaEvent hardware timestamp\n\n");
    printf("  [1]  cuFileDriverOpen      [CPU]: %8.3f ms\n", time_driverOpen       * 1e3);
    printf("  [2]  cudaMalloc (4KB)      [GPU]: %8.3f ms\n", gpu_ms_malloc);
    printf("  [3]  cudaMemset (4KB fill) [GPU]: %8.3f ms\n", gpu_ms_memset);
    printf("  [4]  open(O_DIRECT)        [CPU]: %8.3f ms\n", time_fileOpen         * 1e3);
    printf("  [5]  cuFileHandleRegister  [CPU]: %8.3f ms\n", time_handleRegister   * 1e3);
    printf("  [6]  I/O loop (see below)  [CPU]: %8.3f ms\n", totalTime             * 1e3);
    printf("  [7]  cudaDeviceSynchronize [GPU]: %8.3f ms\n", gpu_ms_sync);
    printf("  [8]  cuFileHandleDeregister[CPU]: %8.3f ms\n", time_handleDeregister * 1e3);
    printf("  [9]  close(fd)             [CPU]: %8.3f ms\n", time_fileClose        * 1e3);
    printf("  [10] cudaFree              [GPU]: %8.3f ms\n", gpu_ms_free);
    printf("  [11] cuFileDriverClose     [CPU]: %8.3f ms\n", time_driverClose      * 1e3);

    printf("\n[GDS Write] 4KB QD=1 I/O Statistics:\n");
    printf("  Total bytes written: %zu (%.2f GB)\n", totalBytesWritten, gbSize);
    printf("  Total I/O time     : %.3f ms\n",        totalTime * 1e3);
    printf("  Throughput         : %.4f GB/s\n",       gbSize / totalTime);
    printf("  IOPS               : %.2f ops/s\n",      iops);
    printf("  Per-block latency [CPU]:\n");
    printf("    Min : %.2f us\n", minLatency_us);
    printf("    Avg : %.2f us\n", avgLat_us);
    printf("    Max : %.2f us\n", maxLatency_us);
    printf("\n");

    return totalTime;
}

// GDS Read: 4KB block size, QD=1 (sequential, one I/O at a time)
double benchmarkGDS(const char* filename, size_t fileSize) {
    const size_t blockSize = 4096;  // 4KB
    const size_t numBlocks = fileSize / blockSize;

    printf("=== GPU Direct Storage Read Benchmark (4KB blocks, QD=1) ===\n");
    printf("  File size : %.2f GB\n", fileSize / (1024.0 * 1024.0 * 1024.0));
    printf("  Block size: %zu B\n",   blockSize);
    printf("  Num blocks: %zu\n\n",   numBlocks);

    // CPU-side wall-clock timers (for non-stream operations)
    double t0, t1;
    double time_driverOpen, time_fileOpen, time_handleRegister,
           time_handleDeregister, time_fileClose, time_driverClose;

    // GPU-side event timers (for CUDA stream operations)
    cudaEvent_t ev_malloc_start, ev_malloc_end;
    cudaEvent_t ev_sync_start,   ev_sync_end;
    cudaEvent_t ev_free_start,   ev_free_end;
    float gpu_ms_malloc, gpu_ms_sync, gpu_ms_free;

    CHECK_CUDA(cudaEventCreate(&ev_malloc_start));
    CHECK_CUDA(cudaEventCreate(&ev_malloc_end));
    CHECK_CUDA(cudaEventCreate(&ev_sync_start));
    CHECK_CUDA(cudaEventCreate(&ev_sync_end));
    CHECK_CUDA(cudaEventCreate(&ev_free_start));
    CHECK_CUDA(cudaEventCreate(&ev_free_end));

    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // [Step 1] Initialize cuFile driver  (CPU-side: pure ioctl, no GPU stream)
    t0 = getTime();
    status = cuFileDriverOpen();
    t1 = getTime();
    time_driverOpen = t1 - t0;
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return -1;
    }

    // [Step 2] Allocate GPU memory (one 4KB buffer, reused for every block)
    //          GPU-side event: cudaMalloc touches GPU MMU
    char* d_buffer;
    CHECK_CUDA(cudaEventRecord(ev_malloc_start, 0));
    CHECK_CUDA(cudaMalloc(&d_buffer, blockSize));
    CHECK_CUDA(cudaEventRecord(ev_malloc_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_malloc_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_malloc, ev_malloc_start, ev_malloc_end));

    // [Step 3] Open file descriptor with O_DIRECT  (CPU-side: VFS/kernel call)
    t0 = getTime();
    int fd = open(filename, O_RDONLY | O_DIRECT);
    t1 = getTime();
    time_fileOpen = t1 - t0;
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT");
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }

    // [Step 4] Register cuFile handle  (CPU-side: ioctl into nvidia-fs)
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    t0 = getTime();
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    t1 = getTime();
    time_handleRegister = t1 - t0;
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }

    // [Step 5] Issue 4KB reads sequentially at QD=1
    //          Each cuFileRead is a synchronous blocking ioctl containing
    //          the full disk I/O + PCIe DMA for that 4KB block.
    //          CPU wall-clock is used per-block; cudaEvent used for the
    //          post-loop synchronize.
    size_t totalBytesRead = 0;
    double minLatency_us = 1e18, maxLatency_us = 0.0, sumLatency_us = 0.0;

    double totalStart = getTime();

    for (size_t i = 0; i < numBlocks; i++) {
        off_t fileOffset = (off_t)(i * blockSize);

        double blockStart = getTime();
        ssize_t ret = cuFileRead(cf_handle, d_buffer, blockSize, fileOffset, 0);
        double blockEnd = getTime();

        if (ret < 0) {
            fprintf(stderr, "cuFileRead failed at block %zu (offset %zu)\n", i, (size_t)fileOffset);
            break;
        }

        totalBytesRead += (size_t)ret;

        double lat_us = (blockEnd - blockStart) * 1e6;
        sumLatency_us += lat_us;
        if (lat_us < minLatency_us) minLatency_us = lat_us;
        if (lat_us > maxLatency_us) maxLatency_us = lat_us;
    }

    // [Step 6] cudaDeviceSynchronize  (GPU-side event: residual GPU work
    //          after all cuFileReads; expected ~0 ms for synchronous GDS)
    CHECK_CUDA(cudaEventRecord(ev_sync_start, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_sync_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_sync_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_sync, ev_sync_start, ev_sync_end));

    double totalTime = getTime() - totalStart;

    // [Step 7] Deregister cuFile handle  (CPU-side: ioctl into nvidia-fs)
    t0 = getTime();
    cuFileHandleDeregister(cf_handle);
    t1 = getTime();
    time_handleDeregister = t1 - t0;

    // [Step 8] Close file descriptor  (CPU-side: VFS)
    t0 = getTime();
    close(fd);
    t1 = getTime();
    time_fileClose = t1 - t0;

    // [Step 9] Free GPU memory  (GPU-side event: cudaFree flushes GPU MMU)
    CHECK_CUDA(cudaEventRecord(ev_free_start, 0));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaEventRecord(ev_free_end, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_free_end));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_free, ev_free_start, ev_free_end));

    // [Step 10] Close cuFile driver  (CPU-side: ioctl)
    t0 = getTime();
    cuFileDriverClose();
    t1 = getTime();
    time_driverClose = t1 - t0;

    // Destroy events
    CHECK_CUDA(cudaEventDestroy(ev_malloc_start));
    CHECK_CUDA(cudaEventDestroy(ev_malloc_end));
    CHECK_CUDA(cudaEventDestroy(ev_sync_start));
    CHECK_CUDA(cudaEventDestroy(ev_sync_end));
    CHECK_CUDA(cudaEventDestroy(ev_free_start));
    CHECK_CUDA(cudaEventDestroy(ev_free_end));

    // --- Report ---
    double gbSize    = (double)totalBytesRead / (1024.0 * 1024.0 * 1024.0);
    double avgLat_us = sumLatency_us / (double)numBlocks;
    double iops      = (double)numBlocks / totalTime;

    printf("[GDS Read] Setup/Teardown Timing:\n");
    printf("  Timer key: [CPU] = wall-clock (CLOCK_MONOTONIC), [GPU] = cudaEvent hardware timestamp\n\n");
    printf("  [1]  cuFileDriverOpen      [CPU]: %8.3f ms\n", time_driverOpen     * 1e3);
    printf("  [2]  cudaMalloc (4KB)      [GPU]: %8.3f ms\n", gpu_ms_malloc);
    printf("  [3]  open(O_DIRECT)        [CPU]: %8.3f ms\n", time_fileOpen       * 1e3);
    printf("  [4]  cuFileHandleRegister  [CPU]: %8.3f ms\n", time_handleRegister * 1e3);
    printf("  [5]  I/O loop (see below)  [CPU]: %8.3f ms\n", totalTime           * 1e3);
    printf("  [6]  cudaDeviceSynchronize [GPU]: %8.3f ms\n", gpu_ms_sync);
    printf("  [7]  cuFileHandleDeregister[CPU]: %8.3f ms\n", time_handleDeregister * 1e3);
    printf("  [8]  close(fd)             [CPU]: %8.3f ms\n", time_fileClose      * 1e3);
    printf("  [9]  cudaFree              [GPU]: %8.3f ms\n", gpu_ms_free);
    printf("  [10] cuFileDriverClose     [CPU]: %8.3f ms\n", time_driverClose    * 1e3);

    printf("\n[GDS Read] 4KB QD=1 I/O Statistics:\n");
    printf("  Total bytes read : %zu (%.2f GB)\n", totalBytesRead, gbSize);
    printf("  Total I/O time   : %.3f ms\n",        totalTime * 1e3);
    printf("  Throughput       : %.4f GB/s\n",       gbSize / totalTime);
    printf("  IOPS             : %.2f ops/s\n",      iops);
    printf("  Per-block latency [CPU]:\n");
    printf("    Min : %.2f us\n", minLatency_us);
    printf("    Avg : %.2f us\n", avgLat_us);
    printf("    Max : %.2f us\n", maxLatency_us);
    printf("\n");

    return totalTime;
}
#endif

int main(int argc, char** argv) {
	// FIle size (1024)^3 -> 1GB
    size_t fileSize = 1024ULL * 1024 * 1024;  // 1GB
    const char* filename = "/mnt/tmp/test_gds.dat";
	const char *writeFilename = "/mnt/tmp/wrtest_gds.dat";
    
    if (argc > 1) {
        fileSize = atoll(argv[1]) * 1024ULL * 1024;  // 1MB scale input arguments
    }

	if (argc > 2) {
		filename = argv[2];
	}
	printf ("FileName : %s\n", filename);
    
    printf("GPU Direct Storage Benchmark\n");
    printf("============================\n");
    printf("File size: %.2f GB\n\n", fileSize / (1024.0*1024.0*1024.0));
    
	// Print GPU Informations
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("PCIe Gen: %d\n", prop.pciDomainID);
    printf("\n");
    
	// Creates Test file in SSD
    createTestFile(filename, fileSize);

#ifdef USE_CUFILE
    // GDS Read: 4KB blocks, QD=1
    benchmarkGDS(filename, fileSize);

    // GDS Write: 4KB blocks, QD=1
    benchmarkGDSWrite(writeFilename, fileSize);
#else
    printf("\nNote: GDS benchmark not available (USE_CUFILE not defined)\n");
    printf("To enable GDS: Install GPUDirect Storage and compile with -DUSE_CUFILE -lcufile\n");
#endif

	// Clean Files
    unlink(filename);
    unlink(writeFilename);
    
    return 0;
}
