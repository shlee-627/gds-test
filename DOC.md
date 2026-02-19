# gds_test.cu – Source Code Workflow Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Build and Usage](#2-build-and-usage)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Structures](#4-data-structures)
5. [Main Program Flow](#5-main-program-flow)
6. [File Setup Workflow](#6-file-setup-workflow)
7. [GDS Benchmark – QD=1 (Synchronous)](#7-gds-benchmark--qd1-synchronous)
8. [GDS Benchmark – QD>1 (Async Sliding Window)](#8-gds-benchmark--qd1-async-sliding-window)
9. [Traditional Benchmark (pread / pwrite)](#9-traditional-benchmark-pread--pwrite)
10. [Statistics Collection and Computation](#10-statistics-collection-and-computation)
11. [Output Workflow](#11-output-workflow)
12. [cuFile API Lifecycle](#12-cufile-api-lifecycle)
13. [Key Design Decisions](#13-key-design-decisions)

---

## 1. Overview

`gds_test.cu` is a random I/O benchmark for **NVIDIA GPU Direct Storage (GDS)**.
GDS allows NVMe storage devices to transfer data directly to and from GPU memory
over PCIe without routing the data through CPU DRAM, eliminating the CPU bounce
buffer that traditional I/O requires.

The program runs two back-to-back benchmark passes on the same file and
configuration:

| Pass | Method | I/O System Call | CPU Memory |
|------|--------|-----------------|------------|
| 1 | **GDS** | `cuFileRead` / `cuFileReadAsync` | GPU buffer via DMA |
| 2 | **Traditional** | `pread` / `pwrite` | CPU buffer via kernel |

This side-by-side comparison shows the latency and bandwidth benefit of GDS
for the exact same workload parameters.

---

## 2. Build and Usage

### Compile

```bash
nvcc -O2 -o gds_test gds_test.cu -lcufile -lm
```

Requires: CUDA >= 11.7, GDS SDK >= 1.3 (`CUFILE_VERSION >= 1030`).

### Command-line Arguments

```
./gds_test --type <RandomRead|RandomWrite>
           --bs   <block_size_bytes>        (default: 4096)
           --iodepth <N>                    (default: 1)
           --file <path>                    (default: /mnt/tmp/gds_bench.dat)
          [--filesize <MB>]                 (default: 1024)
          [--duration <sec>]                (default: 10)
          [--warmup   <sec>]                (default: 2)
          [--create]                        create/overwrite file before test
          [--csv <out.csv>]                 append results to CSV file
          [--ref-bw <GB/s>]                 PCIe reference BW for util% (default: 64.0)
```

### Example Invocations

```bash
# Latency profile: 4 KB random reads, single queue depth
./gds_test --type RandomRead  --bs 4096 --iodepth 1   --file /mnt/nvme/bench.dat --create

# Bandwidth saturation: 4 KB random reads, QD=128
./gds_test --type RandomRead  --bs 4096 --iodepth 128 --file /mnt/nvme/bench.dat \
           --duration 30 --warmup 5 --csv results.csv

# Write bandwidth: 128 KB blocks, QD=32
./gds_test --type RandomWrite --bs 131072 --iodepth 32 --file /mnt/nvme/bench.dat \
           --filesize 4096 --duration 30 --csv results.csv
```

### Argument Validation (parse_args, line 580)

After parsing, the following checks are enforced before any I/O:

- `--bs` must be > 0 and a **multiple of 512** (O_DIRECT kernel requirement)
- `--iodepth` must be >= 1
- `--ref-bw` must be > 0
- `--filesize` must be >= `--bs`
- `file_size` is trimmed down to `floor(file_size / block_size) * block_size`

---

## 3. Architecture Overview

```
main()
  │
  ├─ parse_args()           parse & validate CLI arguments
  ├─ cudaGetDeviceProperties() print GPU / PCIe bus info
  ├─ [--create] create_test_file()
  │
  ├─ run_gds()              GDS benchmark pass
  │    ├─ cuFileDriverOpen()
  │    ├─ open(O_DIRECT)
  │    ├─ drop_page_cache()          (reads only)
  │    ├─ cuFileHandleRegister()
  │    │
  │    ├─ [QD=1]  synchronous loop
  │    │    cudaMalloc → cuFileBufRegister
  │    │    loop: cuFileRead/Write → record latency
  │    │    cuFileBufDeregister → cudaFree
  │    │
  │    └─ [QD>1]  async sliding-window loop
  │         allocate IOSlot[QD], per slot: cudaMalloc + cuFileBufRegister + cudaStreamCreate
  │         initial fill: submit QD async ops
  │         loop: cudaStreamSynchronize → record → resubmit
  │         drain, cuFileBufDeregister, cudaFree, cudaStreamDestroy
  │
  ├─ bench_stats_compute()  sort latencies, compute percentiles, BW, IOPS
  │
  ├─ run_traditional()      Traditional benchmark pass (pread/pwrite)
  │    open(O_DIRECT) → posix_memalign(512) → loop: pread/pwrite → record
  │
  ├─ bench_stats_compute()
  │
  └─ Output: print_stats() + write_csv_row()
```

---

## 4. Data Structures

### 4.1 BenchConfig (line 79)

Holds all benchmark parameters parsed from the command line.

```c
typedef struct {
    IOType  io_type;       // IO_RANDOM_READ or IO_RANDOM_WRITE
    size_t  block_size;    // I/O size in bytes (must be multiple of 512)
    int     io_depth;      // queue depth: 1 = sync path, >1 = async path
    char    filepath[];    // target file path
    size_t  file_size;     // file size in bytes (trimmed to block_size multiple)
    double  duration_s;    // measurement window length (seconds)
    double  warmup_s;      // warmup duration; these ops are discarded from stats
    int     create_file;   // 1 = create/overwrite file before benchmark
    char    csv_path[];    // output CSV path; empty = print to stdout
    double  ref_bw_gbps;   // reference bandwidth for PCIe utilisation %
} BenchConfig;
```

### 4.2 IOSlot (line 100)

One slot in the async sliding window. There are exactly `io_depth` slots.

```
┌──────────────────────────────────────────────────────────────────┐
│  IOSlot                                                          │
│                                                                  │
│  stream      CUDA stream – one per slot, operations on the same  │
│              stream are serialised; different slots run in       │
│              parallel on the NVMe controller                     │
│                                                                  │
│  d_buf       GPU device buffer (cudaMalloc'd, cuFileBufRegister) │
│              Target of the DMA write (read) from NVMe           │
│                                                                  │
│  h_buf       CPU host buffer (posix_memalign, 512-byte aligned)  │
│              Used only by the Traditional path (not GDS)         │
│                                                                  │
│  ─── async cuFile op parameters (pointers passed to driver) ─── │
│  op_size     transfer size (= block_size, constant)              │
│  file_offset random file byte offset, re-drawn before each submit│
│  buf_offset  always 0 (base of d_buf)                           │
│  bytes_done  written by the driver AFTER stream sync             │
│                                                                  │
│  submit_wall CLOCK_MONOTONIC (s) at submission time             │
│  in_flight   1 = async op is outstanding on this slot           │
└──────────────────────────────────────────────────────────────────┘
```

**Why all op parameters are struct members:** `cuFileReadAsync` / `cuFileWriteAsync`
do not copy their arguments at submission time. The driver reads `op_size`,
`file_offset`, and `buf_offset`, and writes `bytes_done`, at an unspecified
point before stream completion. All pointers passed to the async API must
remain valid until `cudaStreamSynchronize` returns.

### 4.3 BenchStats (line 121)

Accumulates raw samples during a benchmark run; derived metrics are computed
afterwards by `bench_stats_compute`.

```c
typedef struct {
    double  *lats_us;       // dynamic latency array (doubles on overflow)
    size_t   lat_n, lat_cap;
    size_t   total_bytes;   // total bytes transferred (post-warmup)
    size_t   total_ops;     // total ops completed    (post-warmup)
    double   t_first;       // completion time of first post-warmup op
    double   t_last;        // completion time of last  post-warmup op
    // derived:
    double   min_us, avg_us, p50_us, p95_us, p99_us, p999_us, max_us;
    double   iops, bw_gbps, wall_time_s;
} BenchStats;
```

---

## 5. Main Program Flow

```
main()
  │
  1. parse_args()       – populate BenchConfig; validate; trim file_size
  2. cudaGetDeviceProperties() – print GPU name and PCIe bus address
  3. print_config()     – display effective benchmark parameters
  │
  4. [if --create]
  │    create_test_file()
  │      └─ posix_fallocate() ──→ success: physical blocks allocated, no write
  │           │ failure
  │           └─ write-zeros loop (4 MB chunks) + fsync()
  │
  5. bench_stats_init(&gds_stats)
  6. run_gds(&cfg, &gds_stats)       ← see §7 / §8
  7. bench_stats_compute(&gds_stats) ← see §10
  │
  8. bench_stats_init(&trad_stats)
  9. run_traditional(&cfg, &trad_stats)   ← see §9
 10. bench_stats_compute(&trad_stats)
  │
 11. print_stats("GDS", ...)
     print_stats("Traditional", ...)
 12. write_csv_row() × 2  →  file (append) or stdout
 13. bench_stats_destroy() × 2
```

---

## 6. File Setup Workflow

### create_test_file (line 227)

The test file must exist on disk with physical blocks allocated before GDS can
DMA into/from it.

```
open(O_WRONLY | O_CREAT | O_TRUNC)
  │
  ├─ posix_fallocate(0, size)
  │     Allocates physical disk blocks without writing data.
  │     Guarantees no ENOSPC during benchmark and prevents sparse-file
  │     short-circuits (sparse reads return zeros from the kernel without
  │     hitting the NVMe).
  │     Supported on most Linux filesystems on NVMe (ext4, xfs, f2fs).
  │
  └─ [fallback if fallocate returns error]
       write() loop in 4 MB chunks filled with zeros
       fsync()   ← ensure data is on disk before O_DIRECT reads
close()
```

### drop_page_cache (line 260)

Called before every read benchmark (both GDS and Traditional) to ensure reads
reach the NVMe device rather than returning cached data.

```
posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED)
  └─ Advises the kernel to evict all cached pages for this file.
     GDS uses O_DIRECT internally (bypasses cache by definition), but
     this call prevents the Traditional read pass from seeing pages warmed
     by the GDS pass or by file creation.

posix_fadvise(fd, 0, size, POSIX_FADV_RANDOM)
  └─ Disables kernel read-ahead prefetching so subsequent reads measure
     individual I/O latency rather than prefetch-assisted throughput.
```

---

## 7. GDS Benchmark – QD=1 (Synchronous)

**Purpose:** Characterise per-operation latency with the highest timer precision.
The CPU brackets each call with `getTime()`, so the measured latency equals the
round-trip time of a single NVMe I/O + PCIe DMA transfer.

### cuFile Object Lifecycle

```
cuFileDriverOpen()           ← initialises nvidia-fs kernel connection
open(O_RDONLY|O_DIRECT)      ← O_DIRECT is mandatory for GDS
drop_page_cache()            ← reads only
cuFileHandleRegister()       ← binds the fd to a cuFile handle

cudaMalloc(d_buf, BS)        ← allocate 4 KB GPU buffer
cuFileBufRegister(d_buf,BS,0)← pin GPU VA in nvidia-fs for DMA
                               (without this, falls back to CPU bounce buffer)

[benchmark loop]

cuFileBufDeregister(d_buf)   ← release GPU DMA pin  (MUST precede cudaFree)
cudaFree(d_buf)

cuFileHandleDeregister()     ← unbind fd           (MUST precede close)
close(fd)
cuFileDriverClose()
```

### Benchmark Loop

```
t_start    = getTime()
t_warm_end = t_start + warmup_s
t_stop     = t_start + warmup_s + duration_s

while getTime() < t_stop:
    offset  = rand_r(seed) % num_blocks * block_size    ← random, aligned
    t_op    = getTime()
    ret     = cuFileRead(cf_handle, d_buf, BS, offset, 0)
              ──── blocking: DMA completes before return ────
    t_done  = getTime()

    if t_op >= t_warm_end:          ← exclude warmup by timestamp
        lat_us = (t_done - t_op) * 1e6
        bench_stats_record(lat_us, ret, t_done)
```

**Timeline (single op):**

```
CPU thread       NVMe / PCIe
─────────────    ──────────────────────────────────────
t_op = getTime()
cuFileRead() ──► ioctl → nvidia-fs → NVMe queue
  (blocked)       NVMe DMA → GPU VRAM (direct, no DRAM hop)
◄─────────────  return bytes_read
t_done = getTime()
```

---

## 8. GDS Benchmark – QD>1 (Async Sliding Window)

**Purpose:** Saturate NVMe bandwidth by keeping `io_depth` operations in flight
simultaneously, analogous to FIO with `--ioengine=libaio`.

### IOSlot Allocation

```
calloc(QD, sizeof(IOSlot))

for s in [0, QD):
    cudaStreamCreate(&slots[s].stream)         ← dedicated stream per slot
    cudaMalloc(&slots[s].d_buf, BS)
    cuFileBufRegister(slots[s].d_buf, BS, 0)  ← mandatory GDS DMA pin
    slots[s].op_size   = BS                    ← constant, pointer passed to async API
    slots[s].buf_offset = 0                    ← constant, pointer passed to async API
    slots[s].in_flight  = 0
```

### Async Submit Helper

Each submission calls:

```c
cuFileReadAsync(cf_handle,
                slot.d_buf,         // GPU buffer base (registered)
                &slot.op_size,      // ─┐
                &slot.file_offset,  //  │ All pointers must stay valid
                &slot.buf_offset,   //  │ until cudaStreamSynchronize
                &slot.bytes_done,   // ─┘ returns for slot.stream
                slot.stream)        // enqueues onto this stream
```

`cuFileReadAsync` returns immediately. The NVMe I/O and PCIe DMA proceed in
the background, ordered by the CUDA stream. `bytes_done` is written by the
driver only after the stream synchronises.

### Sliding-Window Loop

The window maintains exactly `QD` in-flight operations at all times using a
round-robin FIFO discipline — the same slot that was submitted earliest is
waited on first.

```
Phase 1 – Initial Fill
──────────────────────
for s in [0, QD):
    slots[s].file_offset = rand_offset()
    slots[s].submit_wall = getTime()
    cuFileReadAsync(..., slots[s].stream)
    slots[s].in_flight = 1
    in_flight++

Phase 2 – Steady State  (while in_flight > 0)
──────────────────────────────────────────────
idx = next_wait % QD        ← FIFO: wait on oldest slot

if !slots[idx].in_flight:   ← skip non-submitted slots (initial fill error)
    next_wait++; continue

cudaStreamSynchronize(slots[idx].stream)
                            ← blocks until DMA for this slot is done
                            ← bytes_done is NOW valid
t_done = getTime()
slots[idx].in_flight = 0
in_flight--
next_wait++

if slots[idx].bytes_done < 0:
    set time_to_stop        ← I/O error: drain but don't resubmit

if slot.submit_wall >= t_warm_end AND bytes_done > 0:
    bench_stats_record(lat_us, bytes_done, t_done)

if t_done >= t_stop:
    set time_to_stop        ← measurement window expired

if NOT time_to_stop:
    slots[idx].file_offset = rand_offset()
    slots[idx].bytes_done  = 0
    slots[idx].submit_wall = getTime()
    cuFileReadAsync(..., slots[idx].stream)  ← resubmit immediately
    slots[idx].in_flight = 1
    in_flight++

Phase 3 – Drain
───────────────
time_to_stop is set; no new submissions.
The while (in_flight > 0) loop continues, waiting for each remaining
slot via cudaStreamSynchronize, recording stats if post-warmup.
Loop exits when in_flight == 0.
```

**Pipeline diagram (QD=3):**

```
Time →

Slot 0:  [submit]──[DMA in flight]──[sync/record]──[submit]──[DMA]──[sync]──...
Slot 1:  [submit]──────[DMA in flight]────[sync/record]──[submit]──────[DMA]──...
Slot 2:  [submit]──────────[DMA in flight]──────[sync/record]──[submit]──...

CPU:     fill,fill,fill   sync(0),sub(0)  sync(1),sub(1)  sync(2),sub(2) ...
```

The CPU is always blocking on exactly one stream synchronise call while the
other `QD-1` streams run concurrently on the NVMe controller.

### Teardown (mandatory ordering)

```
for s in [0, QD):
    cuFileBufDeregister(slots[s].d_buf)  ← MUST precede cudaFree
    cudaFree(slots[s].d_buf)
    cudaStreamDestroy(slots[s].stream)
free(slots)

cuFileHandleDeregister(cf_handle)        ← MUST precede close(fd)
close(fd)
cuFileDriverClose()
```

---

## 9. Traditional Benchmark (pread / pwrite)

The Traditional path replaces the GDS DMA with a kernel-mediated copy through
CPU DRAM. It runs synchronously regardless of `--iodepth`, demonstrating the
baseline performance without GDS.

### Setup

```
open(O_RDONLY | O_DIRECT)   ← O_DIRECT bypasses page cache at VFS layer
drop_page_cache()            ← reads only; ensures cold-cache measurement
posix_memalign(&h_buf, 512, BS)
                             ← 512-byte alignment required by O_DIRECT for
                                both buffer address and file offset
memset(h_buf, 0xAB, BS)
```

### Benchmark Loop

```
while getTime() < t_stop:
    offset = rand_r(seed) % num_blocks * block_size
    t_op   = getTime()
    ret    = pread(fd, h_buf, BS, offset)
    t_done = getTime()

    if t_op >= t_warm_end:
        bench_stats_record(lat_us, ret, t_done)
```

**Data path comparison:**

```
GDS path (run_gds):
  NVMe ──PCIe DMA──► GPU VRAM
  No CPU DRAM involvement. Latency ≈ NVMe + PCIe transfer time.

Traditional path (run_traditional):
  NVMe ──► kernel page cache ──► CPU DRAM (h_buf) ──► [GPU copy, if needed]
  Extra hop through CPU memory controller adds latency and consumes CPU BW.
```

---

## 10. Statistics Collection and Computation

### Recording (bench_stats_record, line 154)

Called once per completed, post-warmup operation:

```
lat_us      = (t_done - t_op) * 1e6     ← microseconds
bytes       = bytes transferred by this op
t_completion = getTime() at op end

→ append lat_us to lats_us[]  (array doubles in capacity on overflow)
→ total_bytes += bytes
→ total_ops++
→ t_first = t_completion  (frozen at first op; never updated again)
→ t_last  = t_completion  (updated every op)
```

### Wall-Time Accuracy

`wall_time_s` is the span from the **first** to the **last** post-warmup op
completion:

```
wall_time_s = t_last - t_first
```

This is intentionally **not** `getTime() - t_start` because:

- Warmup ops run before `t_first`, so their time is excluded.
- Drain time (the final `QD` ops that complete after `t_stop`) is excluded from
  the denominator but the bytes they transfer are counted, matching how FIO
  reports bandwidth.

### Percentile Computation (bench_stats_compute, line 183)

```
1. qsort(lats_us, lat_n)          ← ascending sort

2. min = lats_us[0]
   max = lats_us[lat_n - 1]
   avg = sum(lats_us) / lat_n

3. Truncating-index percentile:
   idx = floor(p/100 * n)         ← e.g. p99: idx = floor(0.99 * n)
   if idx >= n: idx = n - 1       ← clamp (handles p=100 exactly)
   return lats_us[idx]

   Percentiles computed: P50, P95, P99, P99.9

4. wall_time_s = t_last - t_first  (0 if total_ops < 2)

5. bw_gbps  = total_bytes / 2^30 / wall_time_s
   iops     = total_ops / wall_time_s
```

### Latency Array Growth

Initial capacity: 2²⁰ = 1,048,576 entries (8 MB). When full, `realloc`
doubles the capacity. For a 30-second run at 100,000 IOPS this reaches
~3 million entries (~24 MB), well within typical system memory.

---

## 11. Output Workflow

### Console Output

```
GPU : NVIDIA H100 SXM5
PCI : 0000:3b:00.0

=== GDS Benchmark Configuration ===
  I/O type    : RandomRead
  Block size  : 4096 B  (4 KB)
  Queue depth : 1
                (synchronous – latency focus)
  File        : /mnt/nvme/bench.dat
  File size   : 1.000 GB
  Duration    : 10.0 s  (+2.0 s warmup)
  Ref BW      : 64.0 GB/s

[GDS] Starting ... (RandomRead, QD=1, duration=10s+2s warmup)
[GDS] Done. 9840 ops completed in 10.001 s.

[Traditional] Starting ...
[Traditional] Done. 7210 ops completed in 10.003 s.

=== Results ===

--- GDS ---
  Ops / Data   : 9840 ops / 38.44 MB
  Wall time    : 10.001 s
  Throughput   : 0.0376 GB/s  |  984.00 IOPS
  PCIe util    : 0.06%  (ref: 64.0 GB/s)
  Latency (us) :
    Min    :   1020.14
    Avg    :   1036.58
    P50    :   1031.22
    P95    :   1068.44
    P99    :   1087.33
    P99.9  :   1103.90
    Max    :   1121.00

--- Traditional ---
  ...
```

### CSV Output (line 696)

One row is written per method (GDS, Traditional) per run. The file is opened
in append mode if it already contains data; otherwise the header is written
first.

**Header:**

```
Method,Operation,Pattern,BlockSize_B,QueueDepth,Iterations,
MinLatency_us,AvgLatency_us,P50_us,P95_us,P99_us,P999_us,MaxLatency_us,
Throughput_GBps,IOPS,BandwidthUtil_pct,TotalTime_s
```

**Column definitions:**

| Column | Source | Notes |
|--------|--------|-------|
| `Method` | `"GDS"` or `"Traditional"` | |
| `Operation` | `"Read"` or `"Write"` | |
| `Pattern` | `"Random"` | fixed |
| `BlockSize_B` | `block_size` | bytes (not KB) |
| `QueueDepth` | `io_depth` | |
| `Iterations` | `total_ops` | actual completed ops, not requested count |
| `MinLatency_us` … `MaxLatency_us` | percentile array | microseconds |
| `Throughput_GBps` | `total_bytes / 2^30 / wall_time_s` | |
| `IOPS` | `total_ops / wall_time_s` | |
| `BandwidthUtil_pct` | `bw_gbps / ref_bw_gbps × 100` | default ref: 64 GB/s |
| `TotalTime_s` | `t_last - t_first` | excludes warmup and drain |

---

## 12. cuFile API Lifecycle

The mandatory call ordering enforced by the code is:

```
cuFileDriverOpen()
  │
  open(O_DIRECT)
    │
    cuFileHandleRegister(fd)
      │
      cudaMalloc(d_buf)
        │
        cuFileBufRegister(d_buf)    ← GPU DMA pin created in nvidia-fs
          │
          [cuFileRead / cuFileReadAsync]   ← I/O using DMA path
          │
        cuFileBufDeregister(d_buf)  ← MUST come before cudaFree
        │
      cudaFree(d_buf)
      │
    cuFileHandleDeregister()        ← MUST come before close(fd)
    │
  close(fd)
  │
cuFileDriverClose()
```

**Why the ordering is strict:**

- `cuFileBufRegister` calls `nvidia_p2p_get_pages()` in the kernel to lock the
  GPU physical pages into the PCIe BAR mapping. If `cudaFree` is called first,
  the GPU MMU teardown races with the DMA engine's address translation.
- `cuFileHandleDeregister` holds an internal reference to `fd`. Closing the fd
  first leaves the driver with a dangling file reference.
- Without `cuFileBufRegister`, `cuFileRead` silently uses an internal CPU
  bounce buffer, defeating the purpose of GDS and producing incorrect
  benchmark results.

---

## 13. Key Design Decisions

### Warmup by Timestamp, Not Op Count

```c
// CORRECT (this code):
if (t_op >= t_warm_end) bench_stats_record(...)

// INCORRECT (common mistake):
if (op_index >= warmup_ops) bench_stats_record(...)
```

Op-count warmup is unreliable because op duration varies. Timestamp-based
warmup ensures the driver's internal caches and the NVMe controller's internal
queues are fully warm for a guaranteed wall-clock period before measurement
begins.

### Round-Robin FIFO Wait (not free-slot scan)

```c
idx = (int)(next_wait % (size_t)QD);
cudaStreamSynchronize(slots[idx].stream);
```

This always waits on the *oldest* outstanding slot, not the *first completed*
slot. Compared to a free-slot scan (polling all streams for `cudaStreamQuery ==
cudaSuccess`):

- Avoids 100% CPU usage from busy-wait polling.
- Maintains constant pipeline depth of exactly QD ops.
- Matches the discipline of `io_getevents(min_nr=1)` in Linux libaio.

### Separate Streams Per Slot

Each `IOSlot` owns one `cudaStream_t`. Operations on the same stream are
serialised by CUDA. Operations on different streams (different slots) run
concurrently. This allows `QD` independent NVMe requests to be in flight
simultaneously on the storage controller.

### posix_fallocate over ftruncate

`ftruncate` creates a sparse file on most Linux filesystems. Reading an
unallocated region returns kernel-synthesised zeros without touching the NVMe,
which would produce artificially high benchmark results. `posix_fallocate`
physically allocates disk blocks, ensuring every random read access hits the
NVMe device.

### `wall_time_s = t_last − t_first`

Using `getTime() - t_start` as the denominator for throughput includes warmup
time (inflating the denominator and under-reporting throughput) and drain time
at QD>1 (adding idle time after the last submission, also under-reporting).
The `t_last - t_first` span measures only the window of actual measured I/O.

### BlockSize_B (not BlockSize_KB) in CSV

The CSV column is labelled `BlockSize_B` and stores the raw byte value
(e.g., `4096` for 4 KB). Earlier versions mislabelled this column as
`BlockSize_KB`, causing confusion when the value was already in bytes.
