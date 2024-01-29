#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <string>


__global__ void gpu_matrix_mult(double *a, double *b, double *c, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < n && row < n)
  {
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
      sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
  
}


int main(int argc, char const *argv[])
{

  int block_size = 128, MATRIX_SIZE = 1024;

  //Create and allocate memory for the matrices in the host (CPU)
  double *a = new double[MATRIX_SIZE * MATRIX_SIZE];
  double *b = new double[MATRIX_SIZE * MATRIX_SIZE];
  double *c = new double[MATRIX_SIZE * MATRIX_SIZE];

  // initialize matrix A
  for (int i = 0; i < MATRIX_SIZE; ++i){
      for (int j = 0; j < MATRIX_SIZE; ++j)
      {
          a[i * MATRIX_SIZE + j] = (double(rand() % 50))/10.0;
      }
  }

  // initialize matrix B
  for (int i = 0; i < MATRIX_SIZE; ++i){
      for (int j = 0; j < MATRIX_SIZE; ++j){
          b[i * MATRIX_SIZE + j] = (double(rand() % 50))/10.0;
      }
  }

  // initialize matrix C
  for (int i = 0; i < MATRIX_SIZE; ++i){
      for (int j = 0; j < MATRIX_SIZE; ++j){
          c[i * MATRIX_SIZE + j] = 0;
      }
  }

  //Create and allocate memory for the matrices in the device (GPU)
  double *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
  cudaMalloc((void **)&d_b, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
  cudaMalloc((void **)&d_c, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);

  //Copy the matrices from the host (CPU) to the device (GPU)
  cudaMemcpy(d_a, a, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);

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
  gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a,d_b,d_c, MATRIX_SIZE);
  //Synchronize threads
  cudaDeviceSynchronize();

  //Terminate time counting
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  //Copy the result matrix from the device (GPU) to the host (CPU)
  cudaMemcpy(c, d_c, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyDeviceToHost);

  //Compute elapsed time on GPU computing
  cudaEventElapsedTime(&naive_gpu_time, start, stop);
  printf("Time elapsed on naive GPU matrix multiplication of %dx%d (%d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, block_size, naive_gpu_time);

  // free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
   
  return 0;
}