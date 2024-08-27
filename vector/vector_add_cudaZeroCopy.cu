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
__global__ void Sum_vec_on_GPU(float *vec1, float *vec2, float *vec3, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size){
        vec3[idx] = vec1[idx] + vec2[idx];
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
    float *h_a, *h_b, *GPU_res,*CPU_res;
    float *d_a, *d_b, *d_c;
    const int DSIZE = 1 << 21;
    size_t nBytes = DSIZE * sizeof(float);

    // allocate zerocopy memory
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void**)&h_a, nBytes, flags);
    cudaHostAlloc((void**)&h_b, nBytes, flags);

    // initi data on host
    InitiData(h_a, DSIZE);
    InitiData(h_b, DSIZE);

    GPU_res = (float*)malloc(nBytes);
    CPU_res = (float*)malloc(nBytes);
    memset(CPU_res, 0, nBytes);
    memset(GPU_res, 0, nBytes);

    //pass the pointer to device
    cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0);
    cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0);
    cudaMalloc((void**)&d_c, nBytes);
    cudaCheckErrors("Alloc zero memory failed");

    // Get the result oon the CPU;
    double iStart, iElaps;
    iStart = timeCount();
    Sum_vec_on_CPU(h_a, h_b, CPU_res, DSIZE);
    iElaps = timeCount() - iStart;
    printf("Time taken on CPU : %f seconds.\n", iElaps);

    // warming up
    dim3 block(512);
    dim3 grid((DSIZE + block.x -1)/ block.x); 
    iStart = timeCount();
    Sum_vec_on_GPU<<<grid, block>>>(d_a, d_b, d_c, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel during warming up : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch warming up failure");

    // launch kernel
    iStart = timeCount();
    Sum_vec_on_GPU<<<grid, block>>>(d_a, d_b, d_c, DSIZE);
    cudaDeviceSynchronize();
    iElaps = timeCount() - iStart;
    printf("Time took on kernel : %f seconds.\n", iElaps);
    cudaCheckErrors("kernel launch failure");
    // copy result from device to host
    cudaMemcpy(GPU_res, d_c, nBytes, cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    // check result
    CheckResult(CPU_res, GPU_res, DSIZE);
    printf("Success!\n"); 
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    free(CPU_res);
    free(GPU_res);
    // cudaFree(d_a);  device fopy memory in the host, not alloc so do not cudafree them
    // cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


