#include <stdio.h>
#include <cuda_runtime.h>

#define DSIZE 8096  // Assuming square matrices of size DSIZE x DSIZE
#define BLOCK_SIZE 32

// Initialize matrices with random values
void setInit(int *matr) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        matr[i] = rand() % 1024;  // Random values between 0 and 1023
    }
}

// Compare results from CPU and GPU
void diff_compare(int *mat_cpu, int *mat_gpu) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (mat_cpu[i] != mat_gpu[i]) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, mat_cpu[i], mat_gpu[i]);
        }
    }
}

// CPU-based matrix transposition
void trans_cpu(int *A, int *B) {
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            B[j * DSIZE + i] = A[i * DSIZE + j];
        }
    }
}

// GPU-based matrix transposition with shared memory
__global__ void matrix_trans_smem_gpu(int *A, int *B) {
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];

    // Global indices
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Thread-local indices
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Load from global memory into shared memory
    if (idx < DSIZE && idy < DSIZE) {
        As[local_y][local_x] = A[idy * DSIZE + idx];
    }

    __syncthreads();  // Ensure all threads have loaded their data into shared memory

    // Transpose within shared memory and write back to global memory
    int transposed_idx = blockIdx.y * blockDim.y + threadIdx.x;  // Transpose indices
    int transposed_idy = blockIdx.x * blockDim.x + threadIdx.y;

    if (transposed_idx < DSIZE && transposed_idy < DSIZE) {
        B[transposed_idy * DSIZE + transposed_idx] = As[local_x][local_y];
    }
}

int main(int argc, char **argv) {
    int *h_A, *h_B, *h_B_gpu;
    int *d_A, *d_B;

    int nBytes = DSIZE * DSIZE * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(nBytes);
    h_B = (int*)malloc(nBytes);
    h_B_gpu = (int*)malloc(nBytes);

    // Initialize matrices A and B
    setInit(h_A);
    trans_cpu(h_A, h_B);

    // Allocate device memory
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);

    // Copy matrix A from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel for matrix transposition on the GPU
    matrix_trans_smem_gpu<<<grid, block>>>(d_A, d_B);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result matrix from device to host
    cudaMemcpy(h_B_gpu, d_B, nBytes, cudaMemcpyDeviceToHost);

    // Compare the CPU and GPU results
    diff_compare(h_B, h_B_gpu);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_B_gpu);

    return 0;
}
