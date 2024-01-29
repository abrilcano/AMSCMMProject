#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define tile_width 32

__global__ void tiled_matrix_mult(double *a,double *b, double *c, int n)
{
    __shared__ double ds_M[tile_width][tile_width];
    __shared__ double ds_N[tile_width][tile_width];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  double Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < (n-1) / tile_width + 1; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    if(Row < n && p * tile_width+tx < n) {
        ds_M[ty][tx] = a[Row*n + p*tile_width+tx];
    }
    else
    {
        ds_M[ty][tx] = 0.0;
    }
    if (p*tile_width+ty < n && Col < n) {
        ds_N[ty][tx] = b[(p*tile_width+ty)*n + Col];
    }
    else
    {
        ds_N[ty][tx] = 0.0;
    }
    __syncthreads();

    if(Row < n && Col < n) {
        for (int i = 0; i < tile_width; ++i)
           Pvalue += ds_M[ty][i] * ds_N[i][tx];
    }
    __syncthreads();
  }
  if (Row < n && Col < n)
    c[Row*n+Col] = Pvalue;
}



int main(){

    int block_size = 128, matrix_size = 16384;

    //Create and allocate memory for the matrices
    double *a, *b, *c;
    cudaMallocManaged((void **)&a, sizeof(double) * matrix_size * matrix_size);
    cudaMallocManaged((void **)&b, sizeof(double) * matrix_size * matrix_size);
    cudaMallocManaged((void **)&c, sizeof(double) * matrix_size * matrix_size);

    // initialize matrix A
    for (int i = 0; i < matrix_size; ++i){
      for (int j = 0; j < matrix_size; ++j){
        a[i * matrix_size + j] = (double(rand() % 50))/10.0;
      }
    }

    // initialize matrix B
    for (int i = 0; i < matrix_size; ++i){
      for (int j = 0; j < matrix_size; ++j){
        b[i * matrix_size + j] = (double(rand() % 50))/10.0;
      }
    }

    // initialize matrix C
    for (int i = 0; i < matrix_size; ++i){
      for (int j = 0; j < matrix_size; ++j){
        c[i * matrix_size + j] = 0;
      }
    }

    float gpu_elapsed_time_ms;

    // create cuda events for timing purposes
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    //Define block size
    unsigned int grid_rows = (matrix_size + block_size - 1) / block_size;
    unsigned int grid_cols = (matrix_size + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    //Launch GPU Kernel
    cudaEventRecord(gpu_start, 0);
    tiled_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, matrix_size);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);

    //Calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start, gpu_stop);
    printf("Time elapsed on matrix multiplication of %dx%d on GPU: %f ms.\n\n",matrix_size, matrix_size, gpu_elapsed_time_ms);

    // free memory 
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}