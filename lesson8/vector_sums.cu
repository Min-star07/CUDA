#include <cuda_runtime.h>
#include<stdio.h>

#define N 16384
#define BLOCK_SIZE 32

// Initialize the vector with random values
void setInit(int *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 1024;
    }
}

// Compute the sum on the CPU
void sum_cpu_res(int *vec, int *sum, int size) {
    *sum = 0;
    for (int i = 0; i < size; i++) {
        *sum += vec[i]; // Sum all the elements of the vector
    }
    printf("%d\n", *sum); // Print CPU sum result
}

// CUDA kernel to compute the sum using shared memory
__global__ void sum_gpu_res(int *in, int *out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Compute global index

    __shared__ int smem[BLOCK_SIZE]; // Declare shared memory for each block
    smem[threadIdx.x] = 0; // Initialize shared memory for each thread
    __syncthreads(); // Ensure all threads have initialized their shared memory

    // Load data from global memory to shared memory, incrementing the index to cover the entire array
    for (int index = idx; index < N; index += blockDim.x * gridDim.x) {
        smem[threadIdx.x] += in[index]; // Accumulate values in shared memory
    }
    __syncthreads(); // Synchronize threads before reduction

    // Perform parallel reduction using shared memory
    int temp = 0;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { // Loop over the reduction steps
        if (threadIdx.x < stride) {
            temp = smem[threadIdx.x] + smem[threadIdx.x + stride]; // Reduce adjacent elements
        }
        __syncthreads(); // Ensure all threads have completed the reduction step
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = temp; // Store the result of the reduction back into shared memory
        }
    }

    // Add the block's final sum to the global sum using atomicAdd
    if (threadIdx.x == 0) {
        atomicAdd(out, smem[0]); // The first thread of each block writes the block's sum to global memory
        // printf("%d, %d, %d\n", idx, blockIdx.x, smem[0]);
    }
}

// Compare the CPU and GPU results
void diff_compare(int res_cpu, int res_gpu) {
    printf("cpu result : %d, gpu result : %d\n", res_cpu, res_gpu); // Print CPU and GPU results
}

int main(int argc, char** argv) {
    int *A_h, *A_d; // Host and device pointers
    int sum_cpu, sum_gpu; // Variables to store results

    int nBytes = N * sizeof(int); // Total size in bytes for N integers

    // Allocate host memory and initialize the vector
    A_h = (int*)malloc(nBytes);
    setInit(A_h, N);
    printf("%d\n", A_h[0]); // Print the first element (optional check)

    // Compute the sum on the CPU
    sum_cpu = 0;
    sum_cpu_res(A_h, &sum_cpu, N);
    printf("%d\n", sum_cpu); // Print the CPU sum result

    // Allocate device memory and copy the input vector from host to device
    cudaMalloc((void**)&A_d, nBytes);
    cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / block.x); // Calculate grid size

    // Allocate memory for GPU result and initialize it to zero
    int *sum_d;
    cudaMalloc((void**)&sum_d, sizeof(int));
    cudaMemcpy(sum_d, &sum_gpu, sizeof(int), cudaMemcpyHostToDevice); // Initialize device sum to zero

    // Launch the kernel to compute the sum on the GPU
    sum_gpu_res<<<grid, block>>>(A_d, sum_d, N);
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Copy the GPU result from device to host
    cudaMemcpy(&sum_gpu, sum_d, sizeof(int), cudaMemcpyDeviceToHost);

    // Compare the CPU and GPU results
    diff_compare(sum_cpu, sum_gpu);

    // Free host and device memory
    free(A_h);
    cudaFree(A_d);
    cudaFree(sum_d);

    return 0;
}
