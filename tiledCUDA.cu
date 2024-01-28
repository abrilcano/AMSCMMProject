
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MATRIX_SIZE 8192
#define TILE_SIZE 32

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int n)
{
    __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  int Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < (n-1) / TILE_WIDTH + 1; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    if(Row < n && p * TILE_WIDTH+tx < n) {
        ds_M[ty][tx] = a[Row*n + p*TILE_WIDTH+tx];
    }
    else
    {
        ds_M[ty][tx] = 0.0;
    }
    if (p*TILE_WIDTH+ty < n && Col < n) {
        ds_N[ty][tx] = b[(p*TILE_WIDTH+ty)*n + Col];
    }
    else
    {
        ds_N[ty][tx] = 0.0;
    }
    __syncthreads();

    if(Row < n && Col < n) {
        for (int i = 0; i < TILE_WIDTH; ++i)
           Pvalue += ds_M[ty][i] * ds_N[i][tx];
    }
    __syncthreads();
  }
  if (Row < n && Col < n)
    c[Row*n+Col] = Pvalue;
}



int main(){

    int block_size;

    //Create and allocate memory for the matrices
    double *a, *b, *c;
    cudaMallocManaged((void **)&a, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMallocManaged((void **)&b, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMallocManaged((void **)&c, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);

    // initialize matrix A
    for (int i = 0; i < MATRIX_SIZE; ++i){
      for (int j = 0; j < MATRIX_SIZE; ++j){
        a[i * MATRIX_SIZE + j] = (double(rand() % 50))/10.0;
      }
    }

    // initialize matrix B
    for (int i = 0; i < MATRIX_SIZE; ++i){
      for (int j = 0; j < MATRIX_SIZE; ++j){
        b[i * MATRIX_SIZE + j] = (double(rand() % 50))/10.0;
      }
    }

    float gpu_elapsed_time_ms;

    // create cuda events for timing purposes
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    //Define block size
    unsigned int grid_rows = (MATRIX_SIZE + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    //Launch GPU Kernel
    cudaEventRecord(gpu_start, 0);
    tiled_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, MATRIX_SIZE);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);

    //Calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start, gpu_stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n",
           MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, gpu_elapsed_time_ms);

    // free memory 
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

