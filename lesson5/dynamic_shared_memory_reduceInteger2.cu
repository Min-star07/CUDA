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
        // vec[i] = rand()/(float)RAND_MAX;
        vec[i] = 1.0;
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
void Sum_vec_on_CPU(float * vec1, float sum, int size){
    sum = 0.0;
    for(int i = 0; i < size; i ++){
        sum += vec1[i];
        
    }
    printf("sum=%.1f\n", sum);
}
// kernal
__global__ void Sum_vec_on_GPU_without_shared_interleaved(float *vec1, float* vec2, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *idata = vec1 + blockDim.x * blockIdx.x;
    
    printf("%d, %d, %d, %.1f, %.1f\n", tid, idx, blockIdx.x, idata[0], vec1[idx]);
    
    if(idx > size) return;
    
    for(int stride = blockDim.x/2; stride >0; stride >>=1 ){
        if(tid < stride){
        idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) vec2[blockIdx.x] = idata[0]; 
    
    
}




// kernal

__global__ void Sum_vec_on_GPU_with_shared_interleaved(float *vec1, float* vec2, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  // Global index
    int step = blockDim.x * gridDim.x;              // Stride for loop to cover all elements
    int tid = threadIdx.x;                            // Thread index within the block
    extern __shared__ float s_array[];                    // Shared memory array
    
    // // Load elements into shared memory, with stride to handle larger arrays
    while(idx < size){
            s_array[tid] = vec1[idx];
            idx += step;
            __syncthreads();
    }
    printf("%d, %d, %d, %.1f, %.1f\n", tid, idx, blockIdx.x, s_array[tid], vec1[idx]);
    float *idata = vec1 + blockDim.x * blockIdx.x;

    if(idx > size) return;
    
    for(int stride = blockDim.x/2; stride >0; stride >>=1 ){
         if(tid < stride){
        idata[tid] += idata[tid + stride];
         }
        __syncthreads();
    }
    if(tid == 0) vec2[blockIdx.x] = idata[0]; 
    
    
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
void CheckResult(float CPU_res, float GPU_res, int size){
    float error_bar = 1e-6;
  
    if((CPU_res - GPU_res ) > error_bar){
        printf("ThE result is diffeence between GPU (%.1f) and CPU (%.1f)\n",  CPU_res,  GPU_res);
        exit(1);
    }

    printf("Result : GPU = %.1f, CPU = %.1f\n", GPU_res, CPU_res);
   
}

void sum_from_GPU(float *sum_part, float sum_all, int size){
    for(int i = 0; i < size; i++){
        sum_all += sum_part[i];
         printf("size = %d; sum_part = %.1f; sum = %.1f\n",size, sum_part[i], sum_all);
    }
    
}

int main(int argc, char** argv){
    // set GPU
    setGPU();
    cudaCheckErrors("set GPU failed");
    // define varilable on host and device
    float *A, CPU_res, *GPU_res_with_shared_memory, *GPU_res_without_shared_memory;
    const int DSIZE = 32;
    size_t nBytes = DSIZE * sizeof(float);

    // warming up
    dim3 block(8);
    dim3 grid((DSIZE + block.x -1)/ block.x); 
    cudaMallocManaged((void **)&A, nBytes);
    // cudaMallocManaged((void **)&CPU_res, sizeof(float));
    cudaMallocManaged((void **)&GPU_res_with_shared_memory, grid.x* sizeof(float));
    cudaMallocManaged((void **)&GPU_res_without_shared_memory, grid.x* sizeof(float));

    // initi data on host
    InitiData(A, DSIZE);

    // Count time;
    // Get the result oon the CPU;
    CPU_res = 0.0;
    double iStart, iElaps;
    iStart = timeCount();
    Sum_vec_on_CPU(A, CPU_res, DSIZE);
    iElaps = timeCount() - iStart;
    printf("Time taken on CPU : %f seconds.\n", iElaps);

    
    iStart = timeCount();
    Sum_vec_on_GPU_without_shared_interleaved<<<grid, block>>>(A , GPU_res_without_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel during warming up : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch warming up failure");

    // memset( GPU_res_without_shared_memory, 0, DSIZE);
    // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_without_shared_interleaved<<<grid, block>>>(A , GPU_res_without_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel without shared memory  but integerleaving: %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");
    float GPU_res_without_shared_memory_sum =0;
    sum_from_GPU(GPU_res_without_shared_memory, GPU_res_without_shared_memory_sum, grid.x);


     // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_with_shared_interleaved<<<grid, block, block.x *sizeof(float)>>>(A , GPU_res_with_shared_memory, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel with shared memory and interleaving : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");
    float GPU_res_with_shared_memory_sum =0;
    sum_from_GPU(GPU_res_with_shared_memory, GPU_res_with_shared_memory_sum,grid.x);

    // check result
     CheckResult(CPU_res, GPU_res_without_shared_memory_sum, DSIZE);
    // CheckResult(CPU_res, GPU_res_with_shared_memory_sum, DSIZE);
    printf("Success!\n"); 
    cudaFreeHost(A);
    // cudaFreeHost(CPU_res);
    cudaFreeHost(GPU_res_without_shared_memory);
    cudaFreeHost(GPU_res_with_shared_memory);

    return 0;
}


