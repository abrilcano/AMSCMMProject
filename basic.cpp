#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
// #include <cuda.h>

#define GPU_MATRIX_SIZE 8192
#define CPU_MATRIX_SIZE 2048

void mm_All(const int n, const double *A, const double *B, double *C)
{
    const int UNROLL = 8;
    int rest = n % (4 * UNROLL);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n - rest; i += 4 * UNROLL)
            {
                for (int r = 0; r < 4 * UNROLL; r += 4)
                {
                    _mm256_storeu_pd(C + i + r + j * n, _mm256_add_pd(_mm256_loadu_pd(C + i + r + j * n), _mm256_mul_pd(_mm256_broadcast_sd(A + k + j * n), _mm256_loadu_pd(B + i + r + k * n))));
                }
            }
            for (int i = n - rest; i < n; i++)
            {
                C[j * n + i] += A[j * n + k] * B[k * n + i];
            }
        }
    }
}

int main()
{

    double *A, *B, *C;
    A = (double *)malloc(sizeof(double) * CPU_MATRIX_SIZE * CPU_MATRIX_SIZE);
    B = (double *)malloc(sizeof(double) * CPU_MATRIX_SIZE * CPU_MATRIX_SIZE);
    C = (double *)malloc(sizeof(double) * CPU_MATRIX_SIZE * CPU_MATRIX_SIZE);

    // initialize matrix A
    for (int i = 0; i < CPU_MATRIX_SIZE; ++i)
    {
        for (int j = 0; j < CPU_MATRIX_SIZE; ++j)
        {
            A[i * CPU_MATRIX_SIZE + j] = (double(rand() % 50)) / 10.0;
            ;
        }
    }
    // initialize matrix B
    for (int i = 0; i < CPU_MATRIX_SIZE; ++i)
    {
        for (int j = 0; j < CPU_MATRIX_SIZE; ++j)
        {
            B[i * CPU_MATRIX_SIZE + j] = (double(rand() % 50)) / 10.0;
            ;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    mm_All(CPU_MATRIX_SIZE, A, B, C);
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Best CPU: " << time.count() / 1000.0 << " ms" << std::endl;

    free(A);
    free(B);
    free(C);

    return 0;
}


