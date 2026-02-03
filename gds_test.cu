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

// GPU -> CPU (DRAM) -> File IO
double benchmarkTraditionalWrite(const char* filename, size_t fileSize) {
    printf("=== Traditional Write Benchmark (GPU → SSD) ===\n");

	// Setup Host DRAM Memory
    char* h_buffer = (char*)malloc(fileSize);
    if (!h_buffer) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

	// GPU Memory Allocation & Create Data on GPU
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, fileSize));
    CHECK_CUDA(cudaMemset(d_buffer, 0xCD, fileSize));  // Fill Dummy Data

    double startTime = getTime();

	// 1. GPU Memory -> CPU Memory (cudaMemcpy)
    double copyStart = getTime();
    CHECK_CUDA(cudaMemcpy(h_buffer, d_buffer, fileSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    double copyTime = getTime() - copyStart;

    printf("GPU → CPU: %.3f seconds (%.2f GB/s)\n",
           copyTime, (fileSize / (1024.0*1024.0*1024.0)) / copyTime);

	// 2. CPU Memory -> SSD (File I/O)
    double ioStart = getTime();
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for writing");
        free(h_buffer);
        cudaFree(d_buffer);
        return -1;
    }

    size_t bytesWritten = fwrite(h_buffer, 1, fileSize, fp);
    fflush(fp);
    fclose(fp);
    double ioTime = getTime() - ioStart;

    printf("CPU → File: %.3f seconds (%.2f GB/s)\n",
           ioTime, (fileSize / (1024.0*1024.0*1024.0)) / ioTime);

    double totalTime = getTime() - startTime;
    printf("Total time: %.3f seconds (%.2f GB/s)\n",
           totalTime, (fileSize / (1024.0*1024.0*1024.0)) / totalTime);
    printf("Bytes written: %zu\n\n", bytesWritten);

    free(h_buffer);
    CHECK_CUDA(cudaFree(d_buffer));

    return totalTime;
}

double benchmarkTraditional(const char* filename, size_t fileSize) {
    printf("=== Traditional I/O Benchmark ===\n");
    
    char* h_buffer = (char*)malloc(fileSize);
    if (!h_buffer) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
	// Allocate GPU Memory Buffer
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, fileSize));
    
    double startTime = getTime();
    
   	// 1. Read File from SSD to Host Memory 
    double ioStart = getTime();
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        free(h_buffer);
        cudaFree(d_buffer);
        return -1;
    }
    
    size_t bytesRead = fread(h_buffer, 1, fileSize, fp);		// Due to it uses fread without O_DIRECT, file system is used.
    fclose(fp);
    double ioTime = getTime() - ioStart;
    
    printf("File → CPU: %.3f seconds (%.2f GB/s)\n", 
           ioTime, (fileSize / (1024.0*1024.0*1024.0)) / ioTime);
    
	// 2. Copy copied data from host memory to GDDR (GPU Memory)
    double copyStart = getTime();
    CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, fileSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    double copyTime = getTime() - copyStart;
    
    printf("CPU → GPU: %.3f seconds (%.2f GB/s)\n", 
           copyTime, (fileSize / (1024.0*1024.0*1024.0)) / copyTime);
    
    double totalTime = getTime() - startTime;
    printf("Total time: %.3f seconds (%.2f GB/s)\n", 
           totalTime, (fileSize / (1024.0*1024.0*1024.0)) / totalTime);
    printf("Bytes read: %zu\n\n", bytesRead);
    
	// Free Memory Space
    free(h_buffer);
    CHECK_CUDA(cudaFree(d_buffer));
    
    return totalTime;
}

// Method 3: Use Pinned Memory (Q. What's Pinned Memory?)
double benchmarkPinned(const char* filename, size_t fileSize) {
    printf("=== Pinned Memory I/O Benchmark ===\n");
    
	// Allocates "Pinned" Host Memory Space
    char* h_buffer;
    CHECK_CUDA(cudaMallocHost(&h_buffer, fileSize));
    
	// Allocates GPU Memory space
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, fileSize));
    
    double startTime = getTime();
    
	// 1. Read file from SSD to pinned CPU memory
    double ioStart = getTime();
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        cudaFreeHost(h_buffer);
        cudaFree(d_buffer);
        return -1;
    }
    
    size_t bytesRead = fread(h_buffer, 1, fileSize, fp);
    fclose(fp);
    double ioTime = getTime() - ioStart;
    
    printf("File → Pinned CPU: %.3f seconds (%.2f GB/s)\n", 
           ioTime, (fileSize / (1024.0*1024.0*1024.0)) / ioTime);
    
	// 2. Copies data from pinned CPU memory to GPU Memory (Asynchronously) -> It means it requires copying! NOT shared Memory
    double copyStart = getTime();
    CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, fileSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    double copyTime = getTime() - copyStart;
    
    printf("Pinned CPU → GPU: %.3f seconds (%.2f GB/s)\n", 
           copyTime, (fileSize / (1024.0*1024.0*1024.0)) / copyTime);
    
    double totalTime = getTime() - startTime;
    printf("Total time: %.3f seconds (%.2f GB/s)\n", 
           totalTime, (fileSize / (1024.0*1024.0*1024.0)) / totalTime);
    printf("Bytes read: %zu\n\n", bytesRead);
    
	// Free memory Space
    CHECK_CUDA(cudaFreeHost(h_buffer));
    CHECK_CUDA(cudaFree(d_buffer));
    
    return totalTime;
}

#ifdef USE_CUFILE
// GPU Direct Storage (Write) GPU -> SSD
double benchmarkGDSWrite(const char* filename, size_t fileSize) {
    printf("=== GPU Direct Storage Write Benchmark (GPU → SSD) ===\n");
    
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    
	// Initialize cuFile API
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return -1;
    }
    
	// GPU Memory Allocation & Create Data in GPU
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, fileSize));
    CHECK_CUDA(cudaMemset(d_buffer, 0xCD, fileSize));  // Fill Dummy Data
    
	// Open File Descriptor (WRONLY and O_DIRECT is needed)
    int fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT for writing");
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }
    
	// Registers cuFile Handler
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }
    
    double startTime = getTime();
    
	// GPU Memory -> SSD (File directly)
    ssize_t bytesWritten = cuFileWrite(cf_handle, d_buffer, fileSize, 0, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    double totalTime = getTime() - startTime;
    
    printf("GPU → File (Direct): %.3f seconds (%.2f GB/s)\n", 
           totalTime, (fileSize / (1024.0*1024.0*1024.0)) / totalTime);
    printf("Bytes written: %zd\n\n", bytesWritten);
    
    cuFileHandleDeregister(cf_handle);
    close(fd);
    CHECK_CUDA(cudaFree(d_buffer));
    cuFileDriverClose();
    
    return totalTime;
}

// Method 2: Use Classic GDS
double benchmarkGDS(const char* filename, size_t fileSize) {
    printf("=== GPU Direct Storage Benchmark ===\n");
    
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    
	// Initializes "cuFileAPI"
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile driver open failed: %d\n", status.err);
        return -1;
    }
    
	// Allocates GPU Memory Space -> For GPU Direct Storage (GDS), aligned Memory space is needed)
    char* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, fileSize));
    
	// Open File Descriptor in OS filesystem
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("Failed to open file with O_DIRECT");
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }
    
	// Registers "cuFile Handle"
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile handle register failed: %d\n", status.err);
        close(fd);
        cudaFree(d_buffer);
        cuFileDriverClose();
        return -1;
    }
    
    double startTime = getTime();
    
	// Read file from SSD to "GPU Memory"
    ssize_t bytesRead = cuFileRead(cf_handle, d_buffer, fileSize, 0, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    double totalTime = getTime() - startTime;
    
    printf("File → GPU (Direct): %.3f seconds (%.2f GB/s)\n", 
           totalTime, (fileSize / (1024.0*1024.0*1024.0)) / totalTime);
    printf("Bytes read: %zd\n\n", bytesRead);
    
	
	// Clear File Desc. and Memory Space
    cuFileHandleDeregister(cf_handle);
    close(fd);
    CHECK_CUDA(cudaFree(d_buffer));
    cuFileDriverClose();
    
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
    
    // Executes benchmarks - Method 1 ~ 3
    double tradTime = benchmarkTraditional(filename, fileSize);

	double tradWriteTime = benchmarkTraditionalWrite(writeFilename, fileSize);

#ifdef USE_CUFILE
    double gdsTime = benchmarkGDS(filename, fileSize);
	double gdsWriteTime = benchmarkGDSWrite(writeFilename, fileSize);
    
	// Compare Results
    printf("=== Performance Summary (Read) ===\n");
    printf("Traditional (SSD->GPU):    %.3f seconds (%.2f GB/s)\n", 
           tradTime, (fileSize / (1024.0*1024.0*1024.0)) / tradTime);
    printf("GDS (SSD->GPU):            %.3f seconds (%.2f GB/s) [%.1fx faster]\n", 
           gdsTime, (fileSize / (1024.0*1024.0*1024.0)) / gdsTime,
           tradTime / gdsTime);
	printf("==================================\n\n");
    printf("=== Performance Summary (Write) ===\n");
    printf("Traditional (GPU->SSD):    %.3f seconds (%.2f GB/s)\n", 
           tradWriteTime, (fileSize / (1024.0*1024.0*1024.0)) / tradWriteTime);
    printf("GDS (GPU->SSD):            %.3f seconds (%.2f GB/s) [%.1fx faster]\n", 
           gdsWriteTime, (fileSize / (1024.0*1024.0*1024.0)) / gdsWriteTime,
           tradWriteTime / gdsWriteTime);
	printf("===================================\n");

#else
    printf("\nNote: GDS benchmark not available (USE_CUFILE not defined)\n");
    printf("To enable GDS: Install GPUDirect Storage and compile with -DUSE_CUFILE -lcufile\n");
#endif

	// Clean File
    unlink(filename);
    
    return 0;
}
