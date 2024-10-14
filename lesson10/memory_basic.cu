#include <cstdio>
#include <cuda_runtime.h>

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

    // Host memory allocation
    int h_a[N], h_b[N], h_c[N];

    // Initialize host arrays with values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device memory pointers
    int *d_a, *d_b, *d_c;

    // Step 1: Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Step 2: Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Step 3: Launch kernel for vector addition
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // To ensure N elements are processed
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Step 4: Copy the result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Step 5: Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
