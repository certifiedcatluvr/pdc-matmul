
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matmul_naive(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);
    
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C = (double*)malloc(size);
    
    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (double)rand()/RAND_MAX;
        h_B[i] = (double)rand()/RAND_MAX;
    }
    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N+15)/16, (N+15)/16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Time: %.4f\n", ms/1000.0);
    printf("N: %d\n", N);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
