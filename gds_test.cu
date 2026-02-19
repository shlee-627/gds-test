/*
 * gds_test.cu – GPU Direct Storage (GDS) Random I/O Benchmark
 *
 * Compile:
 *   nvcc -O2 -o gds_test gds_test.cu -lcufile -lm
 *
 * Usage:
 *   ./gds_test --type <RandomRead|RandomWrite>
 *              --bs   <block_size_bytes>          (default: 4096)
 *              --iodepth <N>                      (default: 1)
 *              --file <path>                      (default: /mnt/tmp/gds_bench.dat)
 *             [--filesize <MB>]                   (default: 1024)
 *             [--duration <sec>]                  (default: 10)
 *             [--warmup   <sec>]                  (default: 2)
 *             [--create]                          create/overwrite file before test
 *             [--csv <out.csv>]                   append results to CSV file
 *             [--ref-bw <GB/s>]                   PCIe reference BW for util% (default: 64.0)
 *
 * Design notes:
 *   - QD=1  → synchronous cuFileRead/Write + CPU wall-clock per-op timing.
 *             Gives the most accurate latency histogram.
 *   - QD>1  → cuFileReadAsync/cuFileWriteAsync with one CUDA stream per slot,
 *             round-robin sliding window (analogous to FIO + libaio).
 *             Requires GDS SDK >= 1.3 (CUDA >= 11.7, CUFILE_VERSION >= 1030).
 *   - Traditional path: pread/pwrite with posix_memalign(512) for O_DIRECT.
 *     Synchronous regardless of QD, showing the baseline without GDS.
 *   - Page cache: posix_fadvise(DONTNEED) is called before every read run.
 *   - Buffer registration: every GPU buffer is registered with cuFileBufRegister
 *     before use and deregistered before cudaFree, ensuring the true GDS DMA
 *     path is exercised rather than the internal CPU bounce-buffer fallback.
 *   - Wall time: measured from first to last *completed* post-warmup op, so
 *     warmup and drain phases are excluded from throughput/IOPS computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <cufile.h>

/* ── Compile-time SDK guard ──────────────────────────────────────────────── */
#if defined(CUFILE_VERSION) && (CUFILE_VERSION < 1030)
#  error "GDS async API (cuFileReadAsync/cuFileWriteAsync) requires cuFile SDK >= 1.3 " \
         "(CUFILE_VERSION >= 1030, CUDA >= 11.7). " \
         "Upgrade your GDS package or remove the async QD>1 path."
#endif

/* ── Error-check macros ──────────────────────────────────────────────────── */
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_CF(call)                                                          \
    do {                                                                        \
        CUfileError_t _s = (call);                                              \
        if (_s.err != CU_FILE_SUCCESS) {                                        \
            fprintf(stderr, "cuFile error at %s:%d – code %d\n",               \
                    __FILE__, __LINE__, (int)_s.err);                           \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ── Types ───────────────────────────────────────────────────────────────── */

typedef enum { IO_RANDOM_READ = 0, IO_RANDOM_WRITE = 1 } IOType;

typedef struct {
    IOType  io_type;
    size_t  block_size;      /* bytes                                          */
    int     io_depth;        /* queue depth; 1 = sync path, >1 = async path    */
    char    filepath[4096];
    size_t  file_size;       /* bytes, trimmed to a multiple of block_size      */
    double  duration_s;      /* measurement window (excludes warmup)            */
    double  warmup_s;        /* ops submitted here are discarded from stats     */
    int     create_file;     /* if 1, create/overwrite file before benchmarking */
    char    csv_path[4096];  /* empty → print CSV to stdout                     */
    double  ref_bw_gbps;     /* reference bandwidth for utilisation %            */
} BenchConfig;

/*
 * IOSlot – one position in the async sliding window.
 *
 * ALL pointer fields below are passed by address to cuFileReadAsync /
 * cuFileWriteAsync and MUST remain valid (not freed, not re-used) until
 * cudaStreamSynchronize() returns for the associated stream.  They are
 * members of the slot struct so they automatically outlive each async op.
 */
typedef struct {
    /* GDS async infrastructure */
    cudaStream_t  stream;        /* one dedicated stream per slot              */
    char         *d_buf;         /* cudaMalloc'd GPU buffer, cuFile-registered */

    /* Traditional (CPU) path */
    char         *h_buf;         /* posix_memalign'd buffer, 512-byte aligned  */

    /* async cuFile op parameters – pointers into these fields are passed to
       cuFileReadAsync / cuFileWriteAsync and read by the driver asynchronously */
    size_t        op_size;       /* = block_size, constant for every op        */
    off_t         file_offset;   /* re-randomised before each submit           */
    off_t         buf_offset;    /* always 0 (buf_base is the start of d_buf)  */
    ssize_t       bytes_done;    /* written by the driver upon stream completion*/

    /* timing */
    double        submit_wall;   /* CLOCK_MONOTONIC (s) at submission time     */
    int           in_flight;     /* 1 if an async op is outstanding            */
} IOSlot;

/* Collected statistics for one benchmark run */
typedef struct {
    double  *lats_us;     /* dynamic array of per-op latencies in microseconds */
    size_t   lat_n;       /* number of entries stored                          */
    size_t   lat_cap;     /* allocated capacity (doubled on overflow)          */
    size_t   total_bytes; /* sum of bytes transferred in the measurement window*/
    size_t   total_ops;   /* number of post-warmup ops completed               */
    double   t_first;     /* wall clock at first measured op completion        */
    double   t_last;      /* wall clock at last  measured op completion        */
    /* derived (populated by bench_stats_compute) */
    double   min_us, avg_us, p50_us, p95_us, p99_us, p999_us, max_us;
    double   iops, bw_gbps, wall_time_s;
} BenchStats;

/* ── Wall-clock timer ────────────────────────────────────────────────────── */

static double getTime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Statistics helpers ──────────────────────────────────────────────────── */

static void bench_stats_init(BenchStats *s) {
    memset(s, 0, sizeof(*s));
    s->lat_cap = 1u << 20;                   /* 8 MB initial allocation        */
    s->lats_us = (double *)malloc(s->lat_cap * sizeof(double));
    if (!s->lats_us) { perror("malloc lats"); exit(1); }
    s->t_first = -1.0;
}

/* Record one completed, post-warmup op.
   t_completion: getTime() value captured right after the op returned/synced. */
static void bench_stats_record(BenchStats *s, double lat_us,
                                size_t bytes, double t_completion) {
    if (s->lat_n == s->lat_cap) {
        s->lat_cap *= 2;
        s->lats_us = (double *)realloc(s->lats_us, s->lat_cap * sizeof(double));
        if (!s->lats_us) { perror("realloc lats"); exit(1); }
    }
    if (!isfinite(lat_us) || lat_us < 0.0) lat_us = 0.0; /* clamp bad samples */
    s->lats_us[s->lat_n++] = lat_us;
    s->total_bytes += bytes;
    s->total_ops++;
    if (s->t_first < 0.0) s->t_first = t_completion; /* freeze at first op    */
    s->t_last = t_completion;                          /* advance on every op  */
}

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Truncating-index percentile: idx = floor(pct/100 * n), clamped to n-1.
   Array must already be sorted ascending. Returns 0 if n == 0. */
static double percentile(const double *sorted, size_t n, double p) {
    if (n == 0) return 0.0;
    size_t idx = (size_t)(p / 100.0 * (double)n);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

static void bench_stats_compute(BenchStats *s) {
    if (s->lat_n == 0) return;
    qsort(s->lats_us, s->lat_n, sizeof(double), cmp_dbl);
    s->min_us  = s->lats_us[0];
    s->max_us  = s->lats_us[s->lat_n - 1];
    double sum = 0.0;
    for (size_t i = 0; i < s->lat_n; i++) sum += s->lats_us[i];
    s->avg_us  = sum / (double)s->lat_n;
    s->p50_us  = percentile(s->lats_us, s->lat_n, 50.0);
    s->p95_us  = percentile(s->lats_us, s->lat_n, 95.0);
    s->p99_us  = percentile(s->lats_us, s->lat_n, 99.0);
    s->p999_us = percentile(s->lats_us, s->lat_n, 99.9);
    /*
     * Wall time: span from the first completed post-warmup op to the last.
     * This excludes warmup, drain overhead, and sliding-window startup cost,
     * giving an accurate denominator for throughput and IOPS.
     */
    s->wall_time_s = (s->total_ops >= 2) ? (s->t_last - s->t_first) : 0.0;
    if (s->wall_time_s > 0.0) {
        s->bw_gbps = (double)s->total_bytes / (1024.0*1024.0*1024.0)
                     / s->wall_time_s;
        s->iops    = (double)s->total_ops / s->wall_time_s;
    }
}

static void bench_stats_destroy(BenchStats *s) {
    free(s->lats_us);
    s->lats_us = NULL;
}

/* ── Random-offset generator ─────────────────────────────────────────────── */

static off_t rand_offset(size_t file_size, size_t block_size, unsigned *seed) {
    size_t num_blocks = file_size / block_size;
    return (off_t)((size_t)rand_r(seed) % num_blocks * block_size);
}

/* ── File creation ───────────────────────────────────────────────────────── */

/*
 * posix_fallocate() pre-allocates disk blocks without writing data; this
 * prevents sparse-file read short-circuits and ENOSPC surprises during I/O.
 * Falls back to writing zeros when fallocate is not supported (e.g., tmpfs).
 */
static void create_test_file(const char *path, size_t size) {
    printf("[setup] Creating test file: %s  (%.3f GB) ...\n",
           path, (double)size / (1024.0*1024.0*1024.0));
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open(create)"); exit(1); }

    if (posix_fallocate(fd, 0, (off_t)size) == 0) {
        printf("[setup] Allocated with posix_fallocate.\n");
    } else {
        fprintf(stderr, "[setup] posix_fallocate failed; writing zeros (slower)...\n");
        const size_t CHUNK = 4u << 20;   /* 4 MB chunks */
        char *buf = (char *)calloc(1, CHUNK);
        if (!buf) { perror("calloc"); exit(1); }
        for (size_t written = 0; written < size; ) {
            size_t n = ((size - written) < CHUNK) ? (size - written) : CHUNK;
            ssize_t r = write(fd, buf, n);
            if (r <= 0) { perror("write"); free(buf); exit(1); }
            written += (size_t)r;
        }
        free(buf);
        fsync(fd);
    }
    close(fd);
    printf("[setup] Done.\n\n");
}

/* ── Page-cache eviction ─────────────────────────────────────────────────── */

/*
 * GDS uses O_DIRECT internally so it always bypasses the page cache.
 * We still call this for both paths so the Traditional read run does not
 * benefit from pages warmed by file creation or a prior GDS run.
 */
static void drop_page_cache(int fd, size_t size) {
    if (posix_fadvise(fd, 0, (off_t)size, POSIX_FADV_DONTNEED) != 0)
        perror("[warn] posix_fadvise(DONTNEED) – read results may be cache-warm");
    posix_fadvise(fd, 0, (off_t)size, POSIX_FADV_RANDOM); /* disable read-ahead */
}

/* ── GDS benchmark ───────────────────────────────────────────────────────── */

/*
 * QD=1:  synchronous cuFileRead / cuFileWrite, CPU wall-clock per op.
 *        Best for latency characterisation.
 *
 * QD>1:  async cuFileReadAsync / cuFileWriteAsync, one CUDA stream per slot,
 *        round-robin sliding window.  Best for bandwidth / IOPS saturation.
 *
 * cuFileBufRegister() is called for every GPU buffer before use and
 * cuFileBufDeregister() before cudaFree(), ensuring the direct DMA path
 * (not the internal CPU bounce-buffer fallback) is exercised.
 */
static int run_gds(const BenchConfig *cfg, BenchStats *stats) {
    const int    QD   = cfg->io_depth;
    const size_t BS   = cfg->block_size;
    const double WARM = cfg->warmup_s;
    const double DUR  = cfg->duration_s;

    /* Open the cuFile driver once for the whole benchmark run. */
    CHECK_CF(cuFileDriverOpen());

    /* O_DIRECT is mandatory for GDS; cuFileHandleRegister will reject the fd
       if the file was not opened with O_DIRECT. */
    int oflags = (cfg->io_type == IO_RANDOM_READ)
                 ? (O_RDONLY | O_DIRECT)
                 : (O_WRONLY | O_DIRECT);
    int fd = open(cfg->filepath, oflags, 0644);
    if (fd < 0) {
        perror("gds: open");
        cuFileDriverClose();
        return -1;
    }
    if (cfg->io_type == IO_RANDOM_READ) drop_page_cache(fd, cfg->file_size);

    /* Register the file descriptor with the cuFile driver. */
    CUfileDescr_t  cf_descr;
    CUfileHandle_t cf_handle;
    memset(&cf_descr, 0, sizeof(cf_descr));
    cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cf_descr.handle.fd = fd;
    CHECK_CF(cuFileHandleRegister(&cf_handle, &cf_descr));

    unsigned seed     = 42;
    double t_start    = getTime();
    double t_warm_end = t_start + WARM;
    double t_stop     = t_start + WARM + DUR;

    /* ────────────────────────────────────────────────────────────────────
     * QD = 1: synchronous path
     * ──────────────────────────────────────────────────────────────────── */
    if (QD == 1) {
        char *d_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, BS));
        /* Register the GPU buffer before any cuFile I/O. Without this call
           the driver silently falls back to a CPU bounce-buffer path.     */
        CHECK_CF(cuFileBufRegister(d_buf, BS, 0));
        if (cfg->io_type == IO_RANDOM_WRITE)
            CHECK_CUDA(cudaMemset(d_buf, 0xAB, BS)); /* seed write buffer   */

        while (1) {
            double t_now = getTime();
            if (t_now >= t_stop) break;

            off_t  offset = rand_offset(cfg->file_size, BS, &seed);
            double t_op   = getTime();
            ssize_t ret;
            if (cfg->io_type == IO_RANDOM_READ)
                ret = cuFileRead (cf_handle, d_buf, BS, offset, 0);
            else
                ret = cuFileWrite(cf_handle, d_buf, BS, offset, 0);
            double t_done = getTime();

            if (ret < 0) {
                fprintf(stderr, "gds QD=1: I/O error %zd at offset %zu\n",
                        ret, (size_t)offset);
                break;
            }
            /* Exclude warmup ops from statistics. */
            if (t_op >= t_warm_end)
                bench_stats_record(stats, (t_done - t_op) * 1e6,
                                   (size_t)ret, t_done);
        }

        /* Deregister BEFORE cudaFree. */
        CHECK_CF(cuFileBufDeregister(d_buf));
        CHECK_CUDA(cudaFree(d_buf));
    }
    /* ────────────────────────────────────────────────────────────────────
     * QD > 1: async sliding-window path
     * ──────────────────────────────────────────────────────────────────── */
    else {
        /* Allocate and initialise QD slots. */
        IOSlot *slots = (IOSlot *)calloc((size_t)QD, sizeof(IOSlot));
        if (!slots) { perror("calloc slots"); exit(1); }

        for (int s = 0; s < QD; s++) {
            CHECK_CUDA(cudaStreamCreate(&slots[s].stream));
            CHECK_CUDA(cudaMalloc(&slots[s].d_buf, BS));
            /* cuFileBufRegister – mandatory for the true GDS DMA path.    */
            CHECK_CF(cuFileBufRegister(slots[s].d_buf, BS, 0));
            if (cfg->io_type == IO_RANDOM_WRITE)
                CHECK_CUDA(cudaMemset(slots[s].d_buf, 0xAB, BS));
            slots[s].op_size    = BS;
            slots[s].buf_offset = 0;   /* always use the buffer's base address */
            slots[s].in_flight  = 0;
        }

        /* Initial fill: prime the pipeline with QD concurrent operations. */
        int in_flight    = 0;
        int initial_err  = 0;
        for (int s = 0; s < QD; s++) {
            slots[s].file_offset = rand_offset(cfg->file_size, BS, &seed);
            slots[s].bytes_done  = 0;
            slots[s].submit_wall = getTime();

            /*
             * cuFileReadAsync / cuFileWriteAsync parameter notes:
             *   fh           – registered file handle
             *   bufPtr_base  – GPU buffer (must be cuFileBufRegister'd)
             *   size_p       – pointer to transfer size; driver reads it
             *                  asynchronously; must stay valid until stream sync
             *   file_offset_p– pointer to file byte offset; same lifetime rule
             *   bufPtr_offset_p – pointer to offset into buf; always &buf_offset (0)
             *   bytes_done_p – output written by driver after stream sync
             *   stream       – the CUDA stream ordering this operation
             */
            CUfileError_t sub;
            if (cfg->io_type == IO_RANDOM_READ)
                sub = cuFileReadAsync(cf_handle, slots[s].d_buf,
                                      &slots[s].op_size,
                                      &slots[s].file_offset,
                                      &slots[s].buf_offset,
                                      &slots[s].bytes_done,
                                      slots[s].stream);
            else
                sub = cuFileWriteAsync(cf_handle, slots[s].d_buf,
                                       &slots[s].op_size,
                                       &slots[s].file_offset,
                                       &slots[s].buf_offset,
                                       &slots[s].bytes_done,
                                       slots[s].stream);
            if (sub.err != CU_FILE_SUCCESS) {
                fprintf(stderr, "gds async: submit error on slot %d: %d\n",
                        s, (int)sub.err);
                initial_err = 1;
                break;
            }
            slots[s].in_flight = 1;
            in_flight++;
        }

        /*
         * Round-robin sliding window.
         *
         * We always wait on slot[next_wait % QD] – the "oldest" outstanding
         * operation in FIFO order.  Once it completes we record stats and
         * immediately resubmit into that slot (if the time budget allows).
         * This maintains a constant pipeline depth of QD concurrent ops,
         * analogous to FIO's libaio with iodepth_batch_complete=1.
         *
         * If the initial fill had an error we enter the loop with
         * time_to_stop=1 so the already-submitted ops are drained without
         * issuing any new ones.
         */
        int    time_to_stop = initial_err;
        size_t next_wait    = 0;

        while (in_flight > 0) {
            int idx = (int)(next_wait % (size_t)QD);

            /* If this slot is not in flight (e.g. initial fill short-circuit),
               skip it.  in_flight > 0 guarantees we will eventually reach a
               live slot before cycling back here, so this cannot busy-loop
               while in_flight > 0.                                            */
            if (!slots[idx].in_flight) {
                next_wait++;
                continue;
            }

            /* Block until the oldest slot's CUDA stream is complete.
               bytes_done is valid only after this call returns.               */
            CHECK_CUDA(cudaStreamSynchronize(slots[idx].stream));
            double t_done = getTime();

            slots[idx].in_flight = 0;
            in_flight--;
            next_wait++;

            /* Completion-time error check. */
            if (slots[idx].bytes_done < 0) {
                fprintf(stderr, "gds async: I/O error on slot %d: %zd\n",
                        idx, slots[idx].bytes_done);
                time_to_stop = 1;
            }

            /* Record stats for post-warmup ops only. */
            if (!time_to_stop
                && slots[idx].submit_wall >= t_warm_end
                && slots[idx].bytes_done > 0) {
                bench_stats_record(stats,
                                   (t_done - slots[idx].submit_wall) * 1e6,
                                   (size_t)slots[idx].bytes_done,
                                   t_done);
            }

            /* Check measurement deadline. */
            if (t_done >= t_stop) time_to_stop = 1;

            /* Resubmit into this slot if the benchmark is still running. */
            if (!time_to_stop) {
                slots[idx].file_offset = rand_offset(cfg->file_size, BS, &seed);
                slots[idx].bytes_done  = 0;
                slots[idx].submit_wall = getTime();

                CUfileError_t sub;
                if (cfg->io_type == IO_RANDOM_READ)
                    sub = cuFileReadAsync(cf_handle, slots[idx].d_buf,
                                          &slots[idx].op_size,
                                          &slots[idx].file_offset,
                                          &slots[idx].buf_offset,
                                          &slots[idx].bytes_done,
                                          slots[idx].stream);
                else
                    sub = cuFileWriteAsync(cf_handle, slots[idx].d_buf,
                                           &slots[idx].op_size,
                                           &slots[idx].file_offset,
                                           &slots[idx].buf_offset,
                                           &slots[idx].bytes_done,
                                           slots[idx].stream);
                if (sub.err != CU_FILE_SUCCESS) {
                    fprintf(stderr, "gds async: resubmit error: %d\n",
                            (int)sub.err);
                    time_to_stop = 1;
                } else {
                    slots[idx].in_flight = 1;
                    in_flight++;
                }
            }
            /* If time_to_stop, we simply do not resubmit; the while condition
               will drain the remaining in-flight slots on subsequent passes.  */
        }

        /* Teardown: deregister BEFORE cudaFree (mandatory ordering). */
        for (int s = 0; s < QD; s++) {
            /* All streams must be idle by here (drained in the loop above). */
            CHECK_CF(cuFileBufDeregister(slots[s].d_buf));
            CHECK_CUDA(cudaFree(slots[s].d_buf));
            CHECK_CUDA(cudaStreamDestroy(slots[s].stream));
        }
        free(slots);
    }

    cuFileHandleDeregister(cf_handle);
    close(fd);
    cuFileDriverClose();
    return 0;
}

/* ── Traditional (pread / pwrite) benchmark ──────────────────────────────── */

/*
 * Uses synchronous pread/pwrite with a 512-byte-aligned CPU buffer (required
 * for O_DIRECT).  Queue depth has no effect on actual concurrency here – the
 * benchmark is always effectively QD=1, demonstrating why GDS async matters.
 */
static int run_traditional(const BenchConfig *cfg, BenchStats *stats) {
    const size_t BS   = cfg->block_size;
    const double WARM = cfg->warmup_s;
    const double DUR  = cfg->duration_s;

    int oflags = (cfg->io_type == IO_RANDOM_READ)
                 ? (O_RDONLY | O_DIRECT)
                 : (O_WRONLY | O_DIRECT);
    int fd = open(cfg->filepath, oflags, 0644);
    if (fd < 0) { perror("traditional: open"); return -1; }
    if (cfg->io_type == IO_RANDOM_READ) drop_page_cache(fd, cfg->file_size);

    /* O_DIRECT requires 512-byte alignment for both buffer and offset. */
    char *h_buf = NULL;
    if (posix_memalign((void **)&h_buf, 512, BS) != 0) {
        perror("posix_memalign"); close(fd); return -1;
    }
    memset(h_buf, 0xAB, BS);

    unsigned seed     = 42;
    double t_start    = getTime();
    double t_warm_end = t_start + WARM;
    double t_stop     = t_start + WARM + DUR;

    while (1) {
        if (getTime() >= t_stop) break;
        off_t  offset = rand_offset(cfg->file_size, BS, &seed);
        double t_op   = getTime();
        ssize_t ret;
        if (cfg->io_type == IO_RANDOM_READ)
            ret = pread (fd, h_buf, BS, offset);
        else
            ret = pwrite(fd, h_buf, BS, offset);
        double t_done = getTime();

        if (ret < 0) { perror("traditional: pread/pwrite"); break; }
        if (t_op >= t_warm_end)
            bench_stats_record(stats, (t_done - t_op) * 1e6,
                               (size_t)ret, t_done);
    }

    free(h_buf);
    close(fd);
    return 0;
}

/* ── Argument parsing ────────────────────────────────────────────────────── */

static void parse_args(int argc, char **argv, BenchConfig *cfg) {
    /* Defaults */
    cfg->io_type     = IO_RANDOM_READ;
    cfg->block_size  = 4096;
    cfg->io_depth    = 1;
    strncpy(cfg->filepath, "/mnt/tmp/gds_bench.dat", sizeof(cfg->filepath) - 1);
    cfg->file_size   = 1024ULL * 1024 * 1024;  /* 1 GB */
    cfg->duration_s  = 10.0;
    cfg->warmup_s    = 2.0;
    cfg->create_file = 0;
    cfg->csv_path[0] = '\0';
    cfg->ref_bw_gbps = 64.0;   /* PCIe Gen4 x16 unidirectional ≈ 64 GB/s */

    static struct option long_opts[] = {
        {"type",     required_argument, 0, 't'},
        {"bs",       required_argument, 0, 'b'},
        {"iodepth",  required_argument, 0, 'd'},
        {"file",     required_argument, 0, 'f'},
        {"filesize", required_argument, 0, 's'},
        {"duration", required_argument, 0, 'u'},
        {"warmup",   required_argument, 0, 'w'},
        {"create",   no_argument,       0, 'c'},
        {"csv",      required_argument, 0, 'o'},
        {"ref-bw",   required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
        case 't':
            if      (!strcmp(optarg, "RandomRead"))  cfg->io_type = IO_RANDOM_READ;
            else if (!strcmp(optarg, "RandomWrite")) cfg->io_type = IO_RANDOM_WRITE;
            else { fprintf(stderr, "Unknown --type: %s\n", optarg); exit(1); }
            break;
        case 'b': cfg->block_size  = (size_t)atoll(optarg);                  break;
        case 'd': cfg->io_depth    = atoi(optarg);                            break;
        case 'f': strncpy(cfg->filepath, optarg, sizeof(cfg->filepath) - 1); break;
        case 's': cfg->file_size   = (size_t)atoll(optarg) * 1024ULL * 1024; break;
        case 'u': cfg->duration_s  = atof(optarg);                            break;
        case 'w': cfg->warmup_s    = atof(optarg);                            break;
        case 'c': cfg->create_file = 1;                                       break;
        case 'o': strncpy(cfg->csv_path, optarg, sizeof(cfg->csv_path) - 1); break;
        case 'r': cfg->ref_bw_gbps = atof(optarg);                           break;
        default:
            fprintf(stderr,
                "Usage: ./gds_test --type <RandomRead|RandomWrite>\n"
                "                  --bs <bytes> --iodepth <N> --file <path>\n"
                "                  [--filesize <MB>] [--duration <sec>]\n"
                "                  [--warmup <sec>] [--create]\n"
                "                  [--csv <out.csv>] [--ref-bw <GB/s>]\n");
            exit(1);
        }
    }

    /* Validation */
    if (cfg->block_size == 0)    { fprintf(stderr, "--bs must be > 0\n");      exit(1); }
    if (cfg->io_depth   <= 0)    { fprintf(stderr, "--iodepth must be >= 1\n"); exit(1); }
    if (cfg->ref_bw_gbps <= 0.0) { fprintf(stderr, "--ref-bw must be > 0\n");  exit(1); }
    if (cfg->file_size < cfg->block_size) {
        fprintf(stderr, "--filesize must be >= --bs\n"); exit(1);
    }
    if (cfg->block_size % 512 != 0) {
        fprintf(stderr, "--bs must be a multiple of 512 (O_DIRECT requirement)\n");
        exit(1);
    }
    /* Trim file size to an exact multiple of block size. */
    cfg->file_size = (cfg->file_size / cfg->block_size) * cfg->block_size;
}

/* ── Output helpers ──────────────────────────────────────────────────────── */

static void print_config(const BenchConfig *cfg) {
    printf("=== GDS Benchmark Configuration ===\n");
    printf("  I/O type    : %s\n",
           cfg->io_type == IO_RANDOM_READ ? "RandomRead" : "RandomWrite");
    printf("  Block size  : %zu B  (%zu KB)\n",
           cfg->block_size, cfg->block_size / 1024);
    printf("  Queue depth : %d\n", cfg->io_depth);
    if (cfg->io_depth == 1)
        printf("                (synchronous – latency focus)\n");
    else
        printf("                (async sliding window – bandwidth focus)\n");
    printf("  File        : %s\n", cfg->filepath);
    printf("  File size   : %.3f GB\n",
           (double)cfg->file_size / (1024.0*1024.0*1024.0));
    printf("  Duration    : %.1f s  (+%.1f s warmup)\n",
           cfg->duration_s, cfg->warmup_s);
    printf("  Ref BW      : %.1f GB/s\n\n", cfg->ref_bw_gbps);
}

static void print_stats(const char *label, const BenchConfig *cfg,
                         const BenchStats *s) {
    printf("--- %s ---\n", label);
    if (s->total_ops == 0) {
        printf("  (no ops completed in measurement window)\n\n");
        return;
    }
    printf("  Ops / Data   : %zu ops / %.2f MB\n",
           s->total_ops, (double)s->total_bytes / (1024.0*1024.0));
    printf("  Wall time    : %.3f s\n", s->wall_time_s);
    printf("  Throughput   : %.4f GB/s  |  %.2f IOPS\n",
           s->bw_gbps, s->iops);
    printf("  PCIe util    : %.2f%%  (ref: %.1f GB/s)\n",
           s->bw_gbps / cfg->ref_bw_gbps * 100.0, cfg->ref_bw_gbps);
    printf("  Latency (us) :\n");
    printf("    Min    : %9.2f\n", s->min_us);
    printf("    Avg    : %9.2f\n", s->avg_us);
    printf("    P50    : %9.2f\n", s->p50_us);
    printf("    P95    : %9.2f\n", s->p95_us);
    printf("    P99    : %9.2f\n", s->p99_us);
    printf("    P99.9  : %9.2f\n", s->p999_us);
    printf("    Max    : %9.2f\n", s->max_us);
    printf("\n");
}

static const char *CSV_HEADER =
    "Method,Operation,Pattern,BlockSize_B,QueueDepth,Iterations,"
    "MinLatency_us,AvgLatency_us,P50_us,P95_us,P99_us,P999_us,MaxLatency_us,"
    "Throughput_GBps,IOPS,BandwidthUtil_pct,TotalTime_s\n";

static void write_csv_row(FILE *f, const char *method, const BenchConfig *cfg,
                           const BenchStats *s) {
    fprintf(f,
        "%s,%s,Random,%zu,%d,%zu,"
        "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
        "%.4f,%.2f,%.2f,%.3f\n",
        method,
        cfg->io_type == IO_RANDOM_READ ? "Read" : "Write",
        cfg->block_size,
        cfg->io_depth,
        s->total_ops,
        s->min_us, s->avg_us, s->p50_us, s->p95_us,
        s->p99_us, s->p999_us, s->max_us,
        s->bw_gbps, s->iops,
        s->bw_gbps / cfg->ref_bw_gbps * 100.0,
        s->wall_time_s);
}

/* Open CSV in append mode, writing the header only if the file is new/empty. */
static FILE *csv_open(const char *path) {
    int file_has_content = 0;
    FILE *probe = fopen(path, "r");
    if (probe) {
        char c;
        file_has_content = (fread(&c, 1, 1, probe) == 1);
        fclose(probe);
    }
    FILE *f = fopen(path, file_has_content ? "a" : "w");
    if (!f) { perror("fopen csv"); return NULL; }
    if (!file_has_content) fprintf(f, "%s", CSV_HEADER);
    return f;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    BenchConfig cfg;
    parse_args(argc, argv, &cfg);

    /* GPU info – print PCIe bus/device/domain instead of the meaningless
       pciDomainID field that the original code mislabelled as "PCIe Gen". */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n", prop.name);
    printf("PCI : %04x:%02x:%02x.0\n\n",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    print_config(&cfg);

    if (cfg.create_file)
        create_test_file(cfg.filepath, cfg.file_size);

    /* ── GDS benchmark ── */
    BenchStats gds_stats;
    bench_stats_init(&gds_stats);
    printf("[GDS] Starting ... (%s, QD=%d, duration=%.0fs+%.0fs warmup)\n",
           cfg.io_type == IO_RANDOM_READ ? "RandomRead" : "RandomWrite",
           cfg.io_depth, cfg.duration_s, cfg.warmup_s);
    if (run_gds(&cfg, &gds_stats) != 0) {
        fprintf(stderr, "GDS benchmark failed.\n");
        bench_stats_destroy(&gds_stats);
        return 1;
    }
    bench_stats_compute(&gds_stats);
    printf("[GDS] Done. %zu ops completed in %.3f s.\n\n",
           gds_stats.total_ops, gds_stats.wall_time_s);

    /* ── Traditional benchmark ── */
    BenchStats trad_stats;
    bench_stats_init(&trad_stats);
    printf("[Traditional] Starting ... (pread/pwrite, same config)\n");
    if (run_traditional(&cfg, &trad_stats) != 0)
        fprintf(stderr, "Traditional benchmark failed (continuing).\n");
    bench_stats_compute(&trad_stats);
    printf("[Traditional] Done. %zu ops completed in %.3f s.\n\n",
           trad_stats.total_ops, trad_stats.wall_time_s);

    /* ── Results ── */
    printf("=== Results ===\n\n");
    print_stats("GDS", &cfg, &gds_stats);
    print_stats("Traditional", &cfg, &trad_stats);

    /* ── CSV ── */
    if (cfg.csv_path[0]) {
        FILE *f = csv_open(cfg.csv_path);
        if (f) {
            write_csv_row(f, "GDS",         &cfg, &gds_stats);
            write_csv_row(f, "Traditional", &cfg, &trad_stats);
            fclose(f);
            printf("Results appended to: %s\n", cfg.csv_path);
        }
    } else {
        printf("=== CSV Output ===\n%s", CSV_HEADER);
        write_csv_row(stdout, "GDS",         &cfg, &gds_stats);
        write_csv_row(stdout, "Traditional", &cfg, &trad_stats);
        printf("==================\n");
    }

    bench_stats_destroy(&gds_stats);
    bench_stats_destroy(&trad_stats);
    return 0;
}
