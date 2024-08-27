#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
// Initilization data on host
void InitiData(float *vec, int size){
    for(int i =0; i < size; i++){
        vec[i] = rand()/(float)RAND_MAX;
        // printf("%d, %.1f\n", i, vec[i]);
    }
}

// sum on host
void Sum_vec_on_CPU(float * vec1, float *vec2, float *vec3, int size){
    for(int i = 0; i < size; i ++){
        vec3[i] = vec1[i] + vec2[i];
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

// CUDA ERROR CHECK
#define cudaCheckErrors(msg){\
    cudaError_t __err = cudaGetLastError();\
    if(__err != cudaSuccess){\
    fprintf(stderr, "Fatal error: %s ===> %s ===>  %s : %d\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
    fprintf(stderr, "FURTHER CHECK *****  ABORTION\n");\
    exit(1);\
};\
}

// kernal
__global__ void Sum_vec_on_GPU_normal(float *vec1, float *vec2, float *vec3, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size){
        vec3[idx] = vec1[idx] + vec2[idx];
        // printf("%d, %d, %d, %d, %d, %.1f\n",  idx, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, vec3[idx]);
    }
    __syncthreads();
}

// kernal grid stride loop
__global__ void Sum_vec_on_GPU_loop(float *vec1, float *vec2, float *vec3, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
   for(int i = idx; i < size; i += stride){
        vec3[i] = vec1[i] + vec2[i];
        // printf("%d, %d, %d, %d, %d, %.1f\n",  idx, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, vec3[idx]);
    }
    __syncthreads();
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

double timeCount(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec *1.e-6);
}

int main(int argc, char** argv){
    // set GPU
    setGPU();
    cudaCheckErrors("set GPU failed");
    // define varilable on host and device
    float *A, *B, *GPU_res, *CPU_res;
    const int DSIZE = 1 << 21;
    size_t nBytes = DSIZE * sizeof(float);
    cudaMallocManaged((void **)&A, nBytes);
    cudaMallocManaged((void **)&B, nBytes);
    cudaMallocManaged((void **)&GPU_res, nBytes);
    cudaMallocManaged((void **)&CPU_res, nBytes);

    // initi data on host
    InitiData(A, DSIZE);
    InitiData(B, DSIZE);
    memset(CPU_res, 0, nBytes);
    memset(GPU_res, 0, nBytes);

    // Count time;
    // Get the result oon the CPU;
    double iStart, iElaps;
    iStart = timeCount();
    Sum_vec_on_CPU(A, B, CPU_res, DSIZE);
    iElaps = timeCount() - iStart;
    printf("Time taken on CPU : %f seconds.\n", iElaps);

    // warming up
    dim3 block(512);
    dim3 grid((DSIZE + block.x -1)/ block.x); 
    iStart = timeCount();
    Sum_vec_on_GPU_normal<<<grid, block>>>(A , B, GPU_res, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel during warming up : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch warming up failure");

    // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_normal<<<grid, block>>>(A , B, GPU_res, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel by normal mode : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");

     // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU_loop<<<grid, block>>>(A , B, GPU_res, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel by srid stride loop mode : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");

    // check result
    CheckResult(CPU_res, GPU_res, DSIZE);
    printf("Success!\n"); 
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(GPU_res);
    cudaFreeHost(CPU_res);

    return 0;
}


