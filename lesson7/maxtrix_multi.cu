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
const int block_size = 2;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

__global__ void mmul(const float *A, const float *B, float *C, int ds){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // column index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // row index
    
    // Check bounds
    if((idx < ds) && (idy < ds)){
        float temp = 0;
        for(int i = 0; i < ds; i++){
            temp += A[idy * ds + i] * B[i * ds + idx];
        }
        C[idy * ds + idx] = temp;
    }
}

int main(int argc, char **argv){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Timing variables
    clock_t t0, t1, t2;
    double t1sum = 0;
    double t2sum = 0;

    // Start timing
    t0 = clock();
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    for(int i = 0; i < DSIZE * DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds, Begin compute \n", t1sum);

    // Allocate device memory and copy input data to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));

    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaCheckErrors("cudaMemcpy H2D failure");

    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", t2sum);

    // Verify result
    cudaCheckErrors("cudaMemcpy D2H failure");
    for(int i = 0; i < DSIZE * DSIZE; i++){
        if(h_C[i] != A_val * B_val * DSIZE) {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
            return -1;
        }
    }

    printf("Success!\n");

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
