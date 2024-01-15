//%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <cuda.h>

#define MATRIX_SIZE 8192
#define CPU_MATRIX_SIZE 1024

__global__ void gpu_matrix_mult(double *a, double *b, double *c, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n && row < n)
  {
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
      sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}


int main(int argc, char const *argv[])
{
  int block_size;
  
  //try different block sizes
  for (block_size = 4; block_size <= 32; block_size *= 2)
  {
    //Create and allocate memory for the matrices
    double *a, *b, *c;
    // Use Unified Memory for matrix allocation
    cudaMallocManaged((void **)&a, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMallocManaged((void **)&b, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMallocManaged((void **)&c, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);

    // initialize matrix A
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
      for (int j = 0; j < MATRIX_SIZE; ++j)
      {
        a[i * MATRIX_SIZE + j] = 2;
      }
    }

    // initialize matrix B
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
      for (int j = 0; j < MATRIX_SIZE; ++j)
      {
        b[i * MATRIX_SIZE + j] = 3;
      }
    }

    float naive_gpu_time;

    //Events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int grid_rows = (MATRIX_SIZE + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    cudaEventRecord(start, 0);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, MATRIX_SIZE);
    //Synchronize threads
    cudaThreadSynchronize();

    //Terminate time counting
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Compute elapsed time on GPU computing
    cudaEventElapsedTime(&naive_gpu_time, start, stop);
    printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, block_size, naive_gpu_time);

    // free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  return 0;
}