#include <cuda_runtime.h>
#include <stdio.h>

#define N 10000           // Size of the array
#define BLOCK_SIZE 32     // Number of threads per block

// Initialize the vector with random values
void setInit(int *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 1024;  // Fill the vector with random numbers between 0 and 1023
    }
}

// CUDA kernel for finding the maximum value using block-wise reduction
__global__ void max_reduce(int *in, int *out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Compute the global thread index
    __shared__ int smem[BLOCK_SIZE];  // Shared memory for block-wise reduction

    smem[threadIdx.x] = 0;  // Initialize shared memory for each thread

    // Load data from global memory to shared memory, processing multiple elements per thread
    while (idx < size) {
        smem[threadIdx.x] = max(in[idx], smem[threadIdx.x]);  // Update the maximum value for each thread
        idx += blockDim.x * gridDim.x;  // Stride over the array by grid size
    }
    __syncthreads();  // Ensure all threads have updated shared memory

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + stride]);  // Compare and reduce
        }
        __syncthreads();  // Ensure all threads complete before the next iteration
    }

    // Write the block's result to global memory
    if (threadIdx.x == 0) {
        out[blockIdx.x] = smem[0];  // The first thread in each block writes the result
    }
}

// Function to print the result array
void printres(int *res, int size) {
    for (int i = 0; i < size; i++) {
        printf("Block %d max: %d\n", i, res[i]);  // Print the max value from each block
    }
}

int main(int argc, char** argv) {
    int *A_h, *A_d, *max_res, *max_res_gpu;  // Host and device pointers

    int nBytes = N * sizeof(int);  // Size of the input array in bytes

    // Allocate host memory
    A_h = (int*)malloc(nBytes);  // For input array
    max_res_gpu = (int*)malloc(((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int));  // To hold block-wise results

    // Initialize the input array
    setInit(A_h, N);

    // Allocate device memory
    cudaMalloc((void**)&A_d, nBytes);  // For input array
    cudaMalloc((void**)&max_res, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int));  // For block-wise maximums

    // Copy input array from host to device
    cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Enough blocks to cover the input array

    // Launch the kernel to find the maximum value
    max_reduce<<<grid, block>>>(A_d, max_res, N);
    cudaDeviceSynchronize();  // Wait for GPU to finish

    // Copy block-wise maximum results from device to host
    cudaMemcpy(max_res_gpu, max_res, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results from each block
    printres(max_res_gpu, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Free allocated memory on host and device
    free(A_h);  // Free host input array
    free(max_res_gpu);  // Free host result array
    cudaFree(A_d);  // Free device input array
    cudaFree(max_res);  // Free device result array

    return 0;
}
