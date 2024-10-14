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

    // Unified memory allocation
    int *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    cudaCheckErrors("cudaMallocManaged failed");

    // Initialize arrays with values
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Prefetch data to the GPU (device 0)
    cudaMemPrefetchAsync(a, size, 0);
    cudaMemPrefetchAsync(b, size, 0);
    cudaMemPrefetchAsync(c, size, 0);
    // cudaError_t cudaMemPrefetchAsync(void *devPtr, size_t count, int dstDevice, cudaStream_t stream = 0);
    /*
    devPtr: A pointer to the memory you want to prefetch. This memory must have been allocated using cudaMallocManaged.
    count: The size (in bytes) of the memory to prefetch.
    dstDevice: The device you want to prefetch the memory to. This can be:
            A GPU device ID (like 0, 1, etc.) to prefetch memory to that particular GPU.
            cudaCpuDeviceId to prefetch the memory to the CPU (host).
    stream: (Optional) A CUDA stream for asynchronous execution. If set to 0, the default stream is used.
    */ 
    cudaCheckErrors("cudaMemPrefetchAsync to GPU failed");

    // Launch kernel for vector addition
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(a, b, c, N);
    cudaCheckErrors("Kernel launch failed");

    // Wait for kernel to complete before accessing results on the host
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel execution failed");

    // Prefetch results back to the CPU
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    cudaCheckErrors("cudaMemPrefetchAsync to CPU failed");

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaCheckErrors("cudaFree failed");

    return 0;
}
