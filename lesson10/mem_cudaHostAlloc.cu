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

    // Pinned host memory allocation with mapping
    int *h_a, *h_b, *h_c;
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc(&h_a, size, flags);  // Pinned memory on the host
    cudaHostAlloc(&h_b, size, flags);
    cudaHostAlloc(&h_c, size, flags);

    // Initialize host arrays with values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Get device pointers to mapped host memory
    int *d_a, *d_b, *d_c;
    cudaHostGetDevicePointer(&d_a, h_a, 0);  // d_a points to h_a
    cudaHostGetDevicePointer(&d_b, h_b, 0);  // d_b points to h_b
    cudaHostGetDevicePointer(&d_c, h_c, 0);  // d_c points to h_c

    // Launch kernel for vector addition
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaCheckErrors("Kernel launch failed");

    // Wait for kernel to complete before accessing results on the host
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel execution failed");

    // Since the host memory is mapped to the device, no need for cudaMemcpy

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free pinned host memory (automatically mapped to the device)
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
