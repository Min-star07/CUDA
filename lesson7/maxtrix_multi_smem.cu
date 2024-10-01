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

const int DSIZE = 4;
const int block_size = 2; // Block size for CUDA

__global__ void mmul(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // column index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // row index

    // Declare cache in shared memory
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    if (idx < ds && idy < ds) {
        float temp = 0;
        for (int i = 0; i < ds / block_size; i++) {
            // Load data into shared memory
            As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
            Bs[threadIdx.y][threadIdx.x] = B[idy * ds + (i * block_size + threadIdx.x)];
            // Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];
            __syncthreads();

            // Perform multiplication for the tile
            for (int k = 0; k < block_size; k++) {
                temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            __syncthreads();
        }

        // Write to global memory
        C[idy * ds + idx] = temp;
    }
}

// CPU matrix multiplication for verification
void mmul_cpu(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_cpu, *d_A, *d_B, *d_C;

    // Timing variables
    clock_t t0, t1, t2, t3;
    double t1sum, t2sum;

    // Start timing
    t0 = clock();
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_C_cpu = new float[DSIZE * DSIZE];

    // Initialize matrices A and B
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = i;
        h_B[i] = i;
        h_C[i] = 0;
        h_C_cpu[i] = 0;
    }

    // Perform CPU matrix multiplication
    mmul_cpu(h_A, h_B, h_C_cpu, DSIZE);

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init and CPU matrix multiplication took %f seconds\n", t1sum);

    // Allocate device memory and copy input data to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Set up block and grid dimensions
    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    // Launch the CUDA kernel
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Kernel launch failure");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("GPU matrix multiplication took %f seconds\n", t2sum);

    // Verify GPU result matches CPU result
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (h_C[i] != h_C_cpu[i]) {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], h_C_cpu[i]);
            return -1;
        }
    }

    printf("Success! GPU and CPU results match.\n");

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
