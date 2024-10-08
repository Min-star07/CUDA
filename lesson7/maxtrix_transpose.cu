#include <stdio.h>
#include <cuda_runtime.h>

#define DSIZE 8096    // Assuming square matrices of size DSIZE x DSIZE
#define BLOCK_SIZE 32 // Block size for CUDA kernel

// Macro to check for CUDA errors
#define CUDA_CHECK(call) {                                                \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
               cudaGetErrorString(err));                                  \
        exit(1);                                                          \
    }                                                                     \
}

// Initialize matrices with random values between 0 and 1023
void setInit(int *matr) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        matr[i] = rand() % 1024;
    }
}

// Compare results from CPU and GPU and exit early if mismatch is found
void diff_compare(int *mat_cpu, int *mat_gpu) {
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (mat_cpu[i] != mat_gpu[i]) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, mat_cpu[i], mat_gpu[i]);
            return;
        }
    }
    printf("CPU and GPU results match.\n");
}

// CPU-based matrix transposition
void trans_cpu(int *A, int *B) {
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            B[j * DSIZE + i] = A[i * DSIZE + j];
        }
    }
}

// GPU-based matrix transposition
__global__ void matrix_trans_smem_gpu(int *A, int *B) {
    // Calculate global indices
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Transpose only if within matrix bounds
    if (idx < DSIZE && idy < DSIZE) {
        B[idx * DSIZE + idy] = A[idy * DSIZE + idx];
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

    // Initialize matrices A with random values
    setInit(h_A);

    // CPU transposition
    trans_cpu(h_A, h_B);

    // Allocate device memory for A and B
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, nBytes));

    // Copy matrix A from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel for matrix transposition on the GPU
    matrix_trans_smem_gpu<<<grid, block>>>(d_A, d_B);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Wait for GPU to finish computation
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result matrix from device to host
    CUDA_CHECK(cudaMemcpy(h_B_gpu, d_B, nBytes, cudaMemcpyDeviceToHost));

    // Compare the CPU and GPU results
    diff_compare(h_B, h_B_gpu);

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_B_gpu);

    return 0;
}
