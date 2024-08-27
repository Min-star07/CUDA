#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
double timeCount(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec *1.e-6);
}
// Initilization data on host
void InitiData(float *vec, int size){
    for(int i =0; i < size; i++){
        vec[i] = rand()/(float)RAND_MAX;
        // printf("%d, %.1f\n", i, vec[i]);
    }
}

// CUDA ERROR CHECK
#define cudaCheckErrors(msg){\
    cudaError_t __err = cudaGetLastError();\
    if(__err != cudaSuccess){\
    fprintf(stderr, "Fatal error: %s ===> %s ===>  %s : %d\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
    fprintf(stderr, "FURTHER CHECK *****  ABORTION\n");\
    exit(1);\
};\
}
// sum on host
void Sum_vec_on_CPU(float * vec1, float *vec2, int size){
    for(int i = 0; i < size; i ++){
        vec2[i] = vec1[i] + 1;
    }
}
// kernal
__global__ void Sum_vec_on_GPU_without_shared(float *vec1, float* vec2, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
   for(int i = idx; i < size; i += stride){
        vec2[i] = vec1[i] + 1;
        // printf("%d, %d, %d, %d, %d, %.1f\n",  idx, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, vec3[idx]);
    }
    __syncthreads();
}

// kernal
__global__ void Sum_vec_on_GPU_with_shared(float *vec1, float* vec2, int size){
   int idx = threadIdx.x + blockDim.x * blockIdx.x;  // Global index
    int stride = blockDim.x * gridDim.x;              // Stride for loop to cover all elements
    int tid = threadIdx.x;                            // Thread index within the block
    __shared__ float s_array[512];                    // Shared memory array

    // Load elements into shared memory, with stride to handle larger arrays
    for (int i = idx; i < size; i += stride) {
        s_array[tid] = vec1[i];
        __syncthreads();  // Ensure all threads have loaded data into shared memory

        // Perform computation using shared memory
        vec2[i] = s_array[tid] + 1.0f;
        // printf("%d, %d, %d, %d, %d\n", i, idx, tid, blockDim.x, blockIdx.x);
        __syncthreads();  // Ensure all threads have completed their computations
    }
}

void setGPU(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount  < 1){
        printf("No CUDA device found, exiting...\n");
        exit(1);
    }
    else{
        for(int i = 0; i < deviceCount; i++ ){
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
             printf("Device %d: %s has compute capability : Major: %d Minor: %d \n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
            // set GPU
            cudaSetDevice(i);
            printf("Set GPU %d.\n", i);
        }
    }
}

// compare the difference between CPU and GPU
void CheckResult(float *CPU_res, float *GPU_res, int size){
    float error_bar = 1e-6;
    for(int i =0; i < size; i++){
        if((CPU_res[i] - GPU_res[i] ) > error_bar){
            printf("ThE result is diffeence between GPU (%d, %.1f) and CPU (%d, %.1f)\n", i, CPU_res[i], i , GPU_res[i]);
            exit(1);
        }
    }
}


int main(int argc, char** argv){
    // set GPU
    setGPU();
    cudaCheckErrors("set GPU failed");
    // define varilable on host and device
    float *A, *CPU_res, *GPU_res_with_shared_memory, *GPU_res_without_shared_memory;
    const int DSIZE = 1<<24;
    size_t nBytes = DSIZE * sizeof(float);
    cudaMallocManaged((void **)&A, nBytes);
    cudaMallocManaged((void **)&CPU_res, nBytes);
    cudaMallocManaged((void **)&GPU_res_with_shared_memory, nBytes);
    cudaMallocManaged((void **)&GPU_res_without_shared_memory, nBytes);

    // initi data on host
    InitiData(A, DSIZE);

    // Count time;
    // Get the result oon the CPU;
    double iStart, iElaps;
    iStart = timeCount();
    Sum_vec_on_CPU(A, CPU_res, DSIZE);
    iElaps = timeCount() - iStart;
    printf("Time taken on CPU : %f seconds.\n", iElaps);

    // warming up
    dim3 block(512);
    dim3 grid((DSIZE + block.x -1)/ block.x); 
    iStart = timeCount();
    Sum_vec_on_GPU_without_shared<<<grid, block>>>(A , GPU_res_with_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel during warming up : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch warming up failure");

    // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_without_shared<<<grid, block>>>(A , GPU_res_without_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel without shared memory : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");

     // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_with_shared<<<grid, block>>>(A , GPU_res_with_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel with shared memory : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");

    // check result
    CheckResult(CPU_res, GPU_res_without_shared_memory, DSIZE);
    CheckResult(CPU_res, GPU_res_with_shared_memory, DSIZE);
    printf("Success!\n"); 
    cudaFreeHost(A);
    cudaFreeHost(CPU_res);
    cudaFreeHost(GPU_res_without_shared_memory);
    cudaFreeHost(GPU_res_with_shared_memory);

    return 0;
}


