#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

// Function to calculate elapsed time
double timeCount(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

// Initialization of data on host
void InitData(float *vec, int size){
    for(int i = 0; i < size; i++){
        vec[i] = 1.0f;  // Simple initialization to 1.0 for easy validation
    }
}

// CUDA ERROR CHECK
#define cudaCheckErrors(msg){\
    cudaError_t __err = cudaGetLastError();\
    if(__err != cudaSuccess){\
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",\
            msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
        exit(1);\
    }\
}

// CPU function to sum the vector
float Sum_vec_on_CPU(float *vec, int size){
    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        sum += vec[i];
    }
    return sum;
}

// CUDA kernel using shared memory for reduction
__global__ void Sum_vec_with_shared(float *vec, float* result, int size){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Load input into shared memory
    sdata[tid] = (idx < size) ? vec[idx] : 0.0f;
    __syncthreads();

    // printf("%d, %d, %.1f, %.1f\n", tid, idx, sdata[tid], vec[idx]);

    // Perform reduction in shared memory
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result from each block to global memory
    if(tid == 0){
        result[blockIdx.x] = sdata[0];
    }
}

// CUDA kernel without shared memory for reduction
__global__ void Sum_vec_without_shared(float *vec, float* result, int size){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("%d, %d, %.1f\n", tid, idx, vec[idx]);
    // Perform reduction in global memory
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride && idx + stride < size){
            vec[idx] += vec[idx + stride];
        }
        __syncthreads();
    }

    // Write the result from each block to global memory
    if(tid == 0){
        result[blockIdx.x] = vec[blockIdx.x * blockDim.x];
    }
    // printf("%d, %d, %.1f\n", tid, idx, vec[idx]);
}

// Function to sum partial results on CPU
float sum_from_GPU(float *sum_part, int size){
    float sum_all = 0.0;
    for(int i = 0; i < size; i++){
        sum_all += sum_part[i];
        //  printf("size = %d; sum_part = %.1f; sum = %.1f\n",size, sum_part[i], sum_all);

    }
    return sum_all;
}

int main(){
    const int DSIZE = 32;  // Size of the vector (1 million elements)
    const int BLOCK_SIZE = 8; // Number of threads per block
    int gridSize = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *A, *partial_sums_shared, *partial_sums_no_shared;
    float result_with_shared, result_without_shared, result_CPU;

    // Allocate unified memory accessible from both CPU and GPU
    cudaMallocManaged(&A, DSIZE * sizeof(float));
    cudaMallocManaged(&partial_sums_shared, gridSize * sizeof(float));
    cudaMallocManaged(&partial_sums_no_shared, gridSize * sizeof(float));

    // Initialize data in unified memory
    InitData(A, DSIZE);

    // CPU computation
    double iStart, iElaps;
    iStart = timeCount();
    result_CPU = Sum_vec_on_CPU(A, DSIZE);
    iElaps = timeCount() - iStart;
    printf("Time taken by CPU: %f seconds.\n", iElaps);
    printf("Sum result from CPU: %.1f\n", result_CPU);

     // Timing the GPU execution with shared memory warmingup
    iStart = timeCount();
    
    // Launch kernel with shared memory for reduction
    Sum_vec_with_shared<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(A, partial_sums_shared, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    cudaCheckErrors("Kernel with shared memory execution failed (warmingup)");

    // Sum the partial results on the CPU
    result_with_shared = sum_from_GPU(partial_sums_shared, gridSize);

    
    printf("Time taken by GPU with shared memory (warmingup): %f seconds.\n", iElaps);
    printf("Sum result from GPU with shared memory (warmingup): %.1f\n", result_with_shared);

    // Timing the GPU execution with shared memory
    iStart = timeCount();

    // Launch kernel with shared memory for reduction
    Sum_vec_with_shared<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(A, partial_sums_shared, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    cudaCheckErrors("Kernel with shared memory execution failed");

    // Sum the partial results on the CPU
    result_with_shared = sum_from_GPU(partial_sums_shared, gridSize);

    
    printf("Time taken by GPU with shared memory: %f seconds.\n", iElaps);
    printf("Sum result from GPU with shared memory: %.1f\n", result_with_shared);

    // Timing the GPU execution without shared memory
    iStart = timeCount();

    // Launch kernel without shared memory for reduction
    Sum_vec_without_shared<<<gridSize, BLOCK_SIZE>>>(A, partial_sums_no_shared, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    cudaCheckErrors("Kernel without shared memory execution failed");

    // Sum the partial results on the CPU
    result_without_shared = sum_from_GPU(partial_sums_no_shared, gridSize);

   
    printf("Time taken by GPU without shared memory: %f seconds.\n", iElaps);
    printf("Sum result from GPU without shared memory: %.1f\n", result_without_shared);

    // Compare the results
    if (fabs(result_CPU - result_with_shared) < 1e-5 && fabs(result_CPU - result_without_shared) < 1e-5) {
        printf("Results match between CPU and GPU computations!\n");
    } else {
        printf("Mismatch between CPU and GPU results!\n");
        printf("CPU result: %.1f, GPU with shared memory result: %.1f, GPU without shared memory result: %.1f\n",
                result_CPU, result_with_shared, result_without_shared);
    }

    // Clean up unified memory
    cudaFree(A);
    cudaFree(partial_sums_shared);
    cudaFree(partial_sums_no_shared);

    return 0;
}
