#include <cstdio>
#include <cuda_runtime.h>

// CUDA error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 100;
    size_t size = N * sizeof(int);

    // Pinned host memory allocation
    int *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);  // Pinned memory on the host
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // Initialize host arrays with values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);  // Allocate memory on the GPU
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy to device failed");

    // Launch kernel for vector addition
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaCheckErrors("Kernel launch failed");

    // Wait for kernel to complete before accessing results on the host
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel execution failed");

    // Copy result back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy to host failed");

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaCheckErrors("cudaFree failed");

    // Free pinned host memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
