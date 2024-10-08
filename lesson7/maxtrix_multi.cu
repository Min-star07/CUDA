#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

// Error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int DSIZE = 4;           // Size of the matrices (DSIZE x DSIZE)
const int block_size = 2;      // Block size (2x2 threads per block)
const float A_val = 3.0f;      // Constant value for matrix A
const float B_val = 2.0f;      // Constant value for matrix B

// Matrix multiplication kernel
__global__ void mmul(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // Column index in C
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // Row index in C
    
    // Check if within matrix bounds
    if (idx < ds && idy < ds) {
        float temp = 0;
        for (int i = 0; i < ds; i++) {
            temp += A[idy * ds + i] * B[i * ds + idx];
        }
        C[idy * ds + idx] = temp;
    }
}

int main(int argc, char **argv) {
    float *h_A, *h_B, *h_C;  // Host pointers
    float *d_A, *d_B, *d_C;  // Device pointers

    // Timing variables
    clock_t t0, t1, t2;
    double t1sum = 0, t2sum = 0;

    // Start initialization timing
    t0 = clock();

    // Allocate host memory
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    // Initialize matrices A and B with constant values
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // End initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Initialization took %f seconds. Starting computation...\n", t1sum);

    // Allocate device memory
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Define block and grid dimensions
    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    // Launch matrix multiplication kernel
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Kernel launch failure");

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // End computation timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Computation took %f seconds.\n", t2sum);

    // Verify the result
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        float expected = A_val * B_val * DSIZE;
        if (h_C[i] != expected) {
            printf("Mismatch at index %d: was %f, expected %f\n", i, h_C[i], expected);
            return -1;
        }
    }

    printf("Success! Matrix multiplication completed correctly.\n");

    // Free host and device memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
