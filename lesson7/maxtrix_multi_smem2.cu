#include <stdio.h>
#include <cuda_runtime.h>

#define M 8
#define N 4
#define DSIZE 8192            // Size of square matrices (DSIZE x DSIZE)
#define BLOCK_SIZE 32         // Block size for shared memory

// Initialize matrices with random values between 0 and 1023
void setInit(int *matr) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        matr[i] = rand() % 1024;
    }
}

// CPU matrix multiplication for comparison
void matrix_multi_cpu(int *A, int *B, int *C) {
    for (int y = 0; y < DSIZE; y++) {
        for (int x = 0; x < DSIZE; x++) {
            int temp = 0;
            for (int step = 0; step < DSIZE; step++) {
                temp += A[y * DSIZE + step] * B[step * DSIZE + x];
            }
            C[y * DSIZE + x] = temp;
        }
    }
}

// Compare results from CPU and GPU
void diff_compare(int *mat_cpu, int *mat_gpu) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (mat_cpu[i] != mat_gpu[i]) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, mat_cpu[i], mat_gpu[i]);
            return;
        }
    }
    printf("Results match!\n");
}

// GPU kernel for matrix multiplication using shared memory
__global__ void matrix_multi_smem_gpu(int *A, int *B, int *C) {
    // Shared memory for submatrices of A and B
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Global indices for the current thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Temporary variable to hold the result for C[idy][idx]
    int temp = 0;

    // Iterate through tiles of A and B
    for (int step = 0; step < (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; step++) {
        // Load data into shared memory
        if (idy < DSIZE && (step * BLOCK_SIZE + threadIdx.x) < DSIZE) {
            As[threadIdx.y][threadIdx.x] = A[idy * DSIZE + (step * BLOCK_SIZE + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;  // Handle edge case
        }

        if (idx < DSIZE && (step * BLOCK_SIZE + threadIdx.y) < DSIZE) {
            Bs[threadIdx.y][threadIdx.x] = B[(step * BLOCK_SIZE + threadIdx.y) * DSIZE + idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;  // Handle edge case
        }

        // Synchronize to ensure shared memory is loaded
        __syncthreads();

        // Perform multiplication of the submatrices
        for (int k = 0; k < BLOCK_SIZE; k++) {
            temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result back to global memory if within bounds
    if (idx < DSIZE && idy < DSIZE) {
        C[idy * DSIZE + idx] = temp;
    }
}

int main(int argc, char** argv) {
    int *h_A, *h_B, *h_C, *h_C_gpu;
    int *d_A, *d_B, *d_C;

    int nBytes = DSIZE * DSIZE * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(nBytes);
    h_B = (int*)malloc(nBytes);
    h_C = (int*)malloc(nBytes);
    h_C_gpu = (int*)malloc(nBytes);

    // Initialize matrices A and B with random values
    setInit(h_A);
    setInit(h_B);

    // Perform CPU matrix multiplication
    matrix_multi_cpu(h_A, h_B, h_C);

    // Allocate device memory
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the GPU kernel for matrix multiplication
    matrix_multi_smem_gpu<<<grid, block>>>(d_A, d_B, d_C);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result matrix C from device to host
    cudaMemcpy(h_C_gpu, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    diff_compare(h_C, h_C_gpu);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    return 0;
}
