# Parallel Matrix-Matrix Multiplication with CUDA

## Overview

This project implements matrix-matrix multiplication using CUDA, a parallel computing platform developed by NVIDIA. The code includes various implementations, allowing a comparison between CPU and GPU versions. Specifically, it includes the following files:

- **basic.cpp**: Contains the CPU matrix multiplication implementation.
- **basicCUDAManual.cu**: Implements matrix multiplication on GPU with manual memory management.
- **basicCUDAUnified.cu**: Implements matrix multiplication on GPU with unified memory management.
- **tiledCUDA.cu**: Implements matrix multiplication on GPU with tiling technique.
- **Comparison.cu**: Compares the performance of the GPU implementations using unified memory management, and tiling.
- **MatrixMultiplication.ipynb**: A Jupyter notebook containing all the implementations and results allowing to use google colab computational power and plot the results.

## Compilation and Execution

### Prerequisites

- NVIDIA GPU with CUDA capability.
- CUDA Toolkit installed.
- C++ compiler (e.g., g++) for CPU implementation.

### Compilation

#### CPU Implementation

```bash
g++ basic.cpp -o basic_cpu -O3 -g -Wall -Wextra -pedantic -fsanitize=address -march=native -ffast-math -fopenmp
```

#### GPU Implementation

```bash
nvcc basicCUDAManual.cu -o basic_cuda_manual
nvcc basicCUDAUnified.cu -o basic_cuda_unified
nvcc tiledCUDA.cu -o tiled_cuda
nvcc Comparison.cu -o comparison_cuda
```

### Execution

#### CPU Implementation

```bash
./basic_cpu
```

#### GPU Implementation

```bash
./basic_cuda_manual
./basic_cuda_unified
./tiled_cuda
./comparison_cuda
```

### Results

| Matrix Size | Technique | Time     |
| ----------- | --------- | -------- |
| 2048        | Naive     | 0.004096 |
| 2048        | Tiled     | 0.002208 |
| 4096        | Naive     | 0.006144 |
| 4096        | Tiled     | 0.004064 |
| 8192        | Naive     | 0.004192 |
| 8192        | Tiled     | 0.002464 |
| 16384       | Naive     | 0.005984 |
| 16384       | Tiled     | 0.002048 |
