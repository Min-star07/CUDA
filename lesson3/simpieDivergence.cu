#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

__global__ void warmingup(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if((tid/warpSize) %2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;

    // printf("tid : %d, c[%d] %f\n", tid, tid, c[tid]);
}
__global__ void mathKernel1(float*c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if(tid%2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
    // printf("tid : %d, c[%d] %f\n", tid, tid, c[tid]);
}
__global__ void mathKernel2(float* c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if((tid/warpSize) %2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
    // printf("tid : %d, c[%d] %f\n", tid, tid, c[tid]);
}

__global__ void mathKernel3(float*c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        a = 100.0f;
    }
    if(!ipred)
    {
        b = 200.0f;
    }
    c[tid] = a + b;
    // printf("tid : %d, c[%d] %f\n", tid, tid, c[tid]);
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec *1.e-6);
}
int main(int argc , char** argv){
    // set up device
    cudaDeviceProp deviceProp;
    int devID = 0;
    cudaError_t cudaStatus = cudaSetDevice(devID);
    if(cudaStatus!= cudaSuccess){
        printf("cudaSetDevice failed!  Error code = %d\n", cudaStatus);
    }else{
        cudaStatus = cudaGetDeviceProperties(&deviceProp, devID);
        printf("Device name: %s\n", deviceProp.name);
    }

    // setup data size
    int size = 64;
    int blocksize = 64;
    if(argc >1)
        blocksize = atoi(argv[1]);
    if(argc >2)
        size = atoi(argv[2]);
    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x -1)/block.x, 1);
    printf("Execution Configration (block %d grid %d)\n", block.x, grid.x);
    // allocate memory on device
    float *d_c;
    cudaMalloc((float**)&d_c, size * sizeof(float));
    // run a warmup kernel to remove overhead and count time
    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmingup<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Warmup <<<%4d, %4d>>> elapsed %f seconds\n", grid.x, block.x, iElaps);

    // run the math kernel
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Kernel1 <<<%4d, %4d>>> elapsed %f seconds\n", grid.x, block.x, iElaps);

    // run the math kernel
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Kernel2 <<<%4d, %4d>>> elapsed %f seconds\n",grid.x, block.x, iElaps);

    // run the math kernel
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Kernel3 <<<%4d, %4d>>> elapsed %f seconds\n", grid.x, block.x, iElaps);

    // free gpu memry
    cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}