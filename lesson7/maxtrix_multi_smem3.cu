#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1000  // Rows of A and C
#define K 500  // Columns of A and Rows of B
#define N 1000  // Columns of B and C
#define BLOCK_SIZE 32  // Correct block size for tiling

__managed__ int A[M * K];       // Managed memory for matrix A
__managed__ int B[K * N];       // Managed memory for matrix B
__managed__ int C_gpu[M * N];   // Managed memory for the GPU result matrix
__managed__ int C_cpu[M * N];   // Managed memory for the CPU result matrix

// Initialize matrix with random values
void setInit(int Mat[], int row, int col) {
    for (int y = 0; y < row; y++) {
        for (int x = 0; x < col; x++) {
            Mat[y * col + x] = rand() % 1024;  // Random values between 0 and 1023
        }
    }
}

// CPU matrix multiplication [M * K] x [K * N] = [M * N]
void matrix_multi_cpu(int* A, int* B, int* C) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            int temp = 0;
            for (int step = 0; step < K; step++) { 
                temp += A[y * K + step] * B[step * N + x];
            }
            C[y * N + x] = temp;
        }
    }
}

// Compare CPU and GPU results
void diff_compare(int* res_cpu, int* res_gpu) {
    for (int i = 0; i < M * N; i++) {
        if (abs(res_cpu[i] - res_gpu[i]) > 1e-6) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, res_cpu[i], res_gpu[i]);
        }
    }
}

// GPU kernel for matrix multiplication using shared memory and tiling
__global__ void matrix_multi_gpu(int* A, int* B, int* C) {
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for a submatrix of A
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for a submatrix of B

    // Calculate thread coordinates in matrix C
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Column of C
    int idy = threadIdx.y + blockIdx.y * blockDim.y;  // Row of C

    int temp = 0;  // Accumulate the partial result

    // Loop over tiles of A and B
    for (int step = 0; step < (K + BLOCK_SIZE -1)  / BLOCK_SIZE; step++) {
        // Load A and B tiles into shared memory, ensuring bounds are respected
        if (idy < M && (step * BLOCK_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[idy * K + step * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        if (idx < N && (step * BLOCK_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(step * BLOCK_SIZE + threadIdx.y) * N + idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();  // Ensure all threads have loaded their tiles before proceeding

        // Perform multiplication for the current tile
        for (int i = 0; i < BLOCK_SIZE; i++) {
            temp += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();  // Ensure all threads have completed multiplication before moving to the next tile
    }

    // Write the result to the global memory
    if (idx < N && idy < M) {
        C[idy * N + idx] = temp;
    }
}

int main(int argc, char** argv) {
    // Initialize matrices A and B with random values
    setInit(A, M, K);
    setInit(B, K, N);

    // Perform CPU matrix multiplication
    matrix_multi_cpu(A, B, C_cpu);

    // Define block and grid sizes for the GPU
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Perform GPU matrix multiplication
    matrix_multi_gpu<<<grid, block>>>(A, B, C_gpu);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Compare results from CPU and GPU
    diff_compare(C_cpu, C_gpu);

    return 0;
}
