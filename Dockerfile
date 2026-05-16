FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \\
    gcc \\
    mpich \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy matplotlib pandas

WORKDIR /app
COPY . .

RUN gcc -O2 -fopenmp -o omp_matmul omp_matmul.c
RUN mpicc -O2 -o mpi_matmul mpi_matmul.c
RUN nvcc -O2 -o cuda_naive cuda_naive.cu
RUN nvcc -O2 -o cuda_tiled cuda_tiled.cu

CMD ["python3", "run_all.py"]
