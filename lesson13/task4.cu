#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

const int N = 1024;    // Array size
const int nTPB = 256;  // Threads per block

// Kernel to perform grid-wide sum reduction
__global__ void cooperative_sum_kernel(int* d_data, int* d_result, int data_size) {
    // Cooperative grid group (allows synchronization across all blocks)
    grid_group grid = this_grid();

    // Shared memory for per-block partial sums
    __shared__ int sdata[nTPB];

    // Global thread ID
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned gridSize = blockDim.x * gridDim.x;

    // Each thread sums part of the data
    int sum = 0;
    for (unsigned i = tid; i < data_size; i += gridSize) {
        sum += d_data[i];
    }

    // Store local sum into shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction within block (using shared memory)
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's result
    if (threadIdx.x == 0) {
        atomicAdd(d_result, sdata[0]);
    }

    // Synchronize all blocks before returning
    grid.sync();
}

int main() {
    int* h_data = (int*)malloc(N * sizeof(int));   // Host input array
    int h_result = 0;                              // Host result

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_data[i] = 1;  // For simplicity, all elements are 1, so the expected sum is N
    }

    // Allocate device memory
    int *d_data, *d_result;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));  // Initialize device result to 0

    // Get device properties to ensure cooperative launch is supported
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.cooperativeLaunch == 0) {
        printf("Cooperative kernel launch is not supported on this device.\n");
        return 0;
    }

    // Set grid and block dimensions
    int numBlocks = (N + nTPB - 1) / nTPB;
    void* kernelArgs[] = {(void*)&d_data, (void*)&d_result, (void*)&N};
    dim3 grid(numBlocks);
    dim3 block(nTPB);

    // Launch cooperative kernel
    cudaLaunchCooperativeKernel((void*)cooperative_sum_kernel, grid, block, kernelArgs, 0, 0);

    // Copy the result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Validate the result
    printf("Result from cooperative kernel: %d (expected: %d)\n", h_result, N);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}
