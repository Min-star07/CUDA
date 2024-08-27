#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable(){
    printf("Get the global variable value from host %.1f\n", devData);
    devData += 10.0f;
}

int main(){
    float hostData = 10.0f;
    // copy globl memory variable to device
    cudaMemcpyToSymbol(devData, &hostData, sizeof(float));
    checkGlobalVariable<<<1, 1>>>();
    // copy global memory back to host
    cudaMemcpyFromSymbol(&hostData, devData, sizeof(float));

    printf("Getting the global variable value after kernl : %.1f\n",hostData);
    return 0;
}