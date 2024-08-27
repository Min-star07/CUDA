#include <cuda_runtime.h>
#include <stdio.h>
#include "../freshman.h"
#define SIZE (1<<18)
struct naiveStruct{
    float a[SIZE];
    float b[SIZE];
};

void checkResult_struct(float* h_res_cpu,struct naiveStruct* h_res_gpu,int nElem)
{
    for(int i=0;i<nElem;i++)
        if (h_res_cpu[i]!=h_res_gpu->a[i])
        {
            printf("check fail!\n");
            exit(0);
        }
    printf("result check success!\n");
}

__global__ void sumArraysOnGPU(float *A, float *B, struct naiveStruct *RESULT, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<size){
        RESULT->a[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv){
    setGPU();
    cudaCheckErrors("Set devices failed!!!");

    // int offset = 0;
    // if(argc >1){
    //     offset = atoi(argv[1]);
    // }
    
    int nElem = SIZE;
    int nBytes = nElem * sizeof(float);
    int nBytes_struct = sizeof(struct naiveStruct);
    printf("%.f kb\n", nBytes_struct/1.0);
    printf("vector size : %.3f MB\n", nBytes/1024.0/1024.0);

    // set host memry and initilization
    float *h_a, *h_b, *h_res_from_cpu;
    struct naiveStruct *h_res_from_gpu;
    h_a = (float*) malloc(nBytes);
    h_b = (float*) malloc(nBytes);
    h_res_from_cpu = (float*) malloc(nBytes_struct);
    h_res_from_gpu = (struct naiveStruct*) malloc(nBytes_struct);

    InitialData(h_a, nElem);
    InitialData(h_b, nElem);
    memset(h_res_from_cpu, 0 , nBytes);
    memset(h_res_from_gpu, 0 , nBytes_struct);

    // set device memory ;
    float *d_a,  *d_b;
    struct naiveStruct *d_res_from_gpu;
    CHECK(cudaMalloc((float**) &d_a, nBytes));
    CHECK(cudaMalloc((float**) &d_b, nBytes));
    CHECK(cudaMalloc((struct naiveStruct**) &d_res_from_gpu, nBytes_struct));

    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));
    // cudaMemcpy(d_res_from_gpu, h_res_from_gpu, nBytes, cudaMemcpyHostToDevice);
    CHECK(cudaMemset(d_res_from_gpu, 0 , nBytes_struct));
    // cudaCheckErrors("Copy data from host to device failed!!!");

    dim3 block(1024);
    dim3 grid((nElem + block.x - 1)/block.x);

    double iStart, iStop;
    iStart = timeCount();
    sumArraysOnGPU<<<grid, block>>>(d_a, d_b, d_res_from_gpu, nElem);
    cudaDeviceSynchronize();
    iStop = timeCount() - iStart;
    printf("Execution configuration <<<%d, %d>>>, time consuming : %f s\n", grid.x, block.x, iStop);
    // cudaCheckErrors("launch kernel failed!!!");

    CHECK(cudaMemcpy(h_res_from_gpu, d_res_from_gpu, nBytes_struct, cudaMemcpyDeviceToHost));
    // cudaCheckErrors("Copy data from device to host failed!!!");

    iStart = timeCount();
    sumArraysOnCPU(h_a, h_b, h_res_from_cpu,nElem);
    iStop = timeCount() - iStart;
    printf("Execution configuration on CPU, time consuming : %f s\n", iStop);

    checkResult_struct(h_res_from_cpu, h_res_from_gpu, nElem);

    free(h_a);
    free(h_b);
    free(h_res_from_cpu);
    free(h_res_from_gpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res_from_gpu);

  return 0;
}