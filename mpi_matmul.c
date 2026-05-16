
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = atoi(argv[1]);
    int rows_per_proc = N / size;
    
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = (double*)malloc(rows_per_proc * N * sizeof(double));
    double *local_C = (double*)malloc(rows_per_proc * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        srand(42);
        for (int i = 0; i < N*N; i++) {
            A[i] = (double)rand()/RAND_MAX;
            B[i] = (double)rand()/RAND_MAX;
        }
    }
    
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_proc*N, MPI_DOUBLE, local_A, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double start = MPI_Wtime();
    
    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += local_A[i*N+k] * B[k*N+j];
            local_C[i*N+j] = sum;
        }
    
    MPI_Gather(local_C, rows_per_proc*N, MPI_DOUBLE, C, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double elapsed = MPI_Wtime() - start;
    
    if (rank == 0) {
        printf("Time: %.4f\n", elapsed);
        printf("Processes: %d\n", size);
        printf("N: %d\n", N);
        free(A); free(C);
    }
    
    free(local_A); free(local_C); free(B);
    MPI_Finalize();
    return 0;
}
