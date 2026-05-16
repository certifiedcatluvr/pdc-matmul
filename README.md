# PDC Matrix Multiplication Framework
**Zainab Ehsan | FA23/BSCS/347**
**Course:** Parallel and Distributed Computing

## Overview
Cloud-enabled distributed matrix multiplication framework combining CUDA, OpenMP, MPI and Docker.

## Implementations
- `cpu_matmul.c` — Sequential CPU baseline
- `cuda_naive.cu` — CUDA naive kernel
- `cuda_tiled.cu` — CUDA tiled shared memory kernel
- `omp_matmul.c` — OpenMP parallelization
- `mpi_matmul.c` — MPI distributed execution
- `Dockerfile` — Container configuration

## Results (Tesla T4 GPU, N=512)
| Method | Time | Speedup |
|--------|------|---------|
| CPU Sequential | 0.4174s | 1.00x |
| OpenMP | 0.3496s | 1.19x |
| MPI (4 procs) | 0.3742s | 1.12x |
| CUDA Naive | 0.0020s | 208.70x |
| CUDA Tiled | 0.0012s | **347.83x** |

## Cloud Environment
Google Colab · Tesla T4 GPU · CUDA 11.8
