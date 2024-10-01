#include <cuda_runtime.h>
#include <stdio.h>
#include "../freshman.h"
#define N_REPEAT 10
#define N_SEGMENT 4
void CUDART_CB my_callback(cudaStream_t stream,cudaError_t status,void * data)
{
    printf("call back from stream:%d\n",*((int *)data));
}
void sumArrays(int * a,int * b,int * res,const int size)
{
    for(int i=0;i<size;i+=4)
    {
        res[i]=a[i]+b[i];
        res[i+1]=a[i+1]+b[i+1];
        res[i+2]=a[i+2]+b[i+2];
        res[i+3]=a[i+3]+b[i+3];
    }
}
__global__ void sumArraysGPU(int*a,int*b,int*res,int N)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < N)
    //for delay
    {
        for(int j=0;j<N_REPEAT;j++)
            res[idx]=a[idx]+b[idx];
    }

}

void checkResult(int * hostRef,int * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%d(hostRef[%d] )!= %d(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}
int main(){
    setGPU();
    double t_start, t_stop;
    t_start = timeCount();
    int nElem = 1 << 24;
    printf("Vector size : %d\n", nElem);
    int nBytes = sizeof(int)*nElem;

    int *h_a, *h_b, *h_res, *h_res_from_gpu;

    CHECK(cudaHostAlloc((int**)&h_a, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((int**)&h_b, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((int**)&h_res, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((int**)&h_res_from_gpu, nBytes, cudaHostAllocDefault));

    cudaMemset(h_res,0, nBytes);
    cudaMemset(h_res_from_gpu, 0,nBytes);

    int *d_a, *d_b, *d_res;
    CHECK(cudaMalloc((int**)&d_a, nBytes));
    CHECK(cudaMalloc((int**)&d_b, nBytes));
    CHECK(cudaMalloc((int**)&d_res, nBytes));

    InitialData(h_a, nElem);
    InitialData(h_b, nElem);

    sumArrays(h_a, h_b, h_res, nElem);

    dim3 block(512);
    dim3 grid((nElem + block.x - 1)/ block.x);

    //asynchronous calcaulation
    int iElem = nElem / N_SEGMENT;
    cudaStream_t stream[N_SEGMENT];
    for(int i =0; i < N_SEGMENT; i++){
        CHECK(cudaStreamCreate(&stream[i]));
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    for(int i =0; i < N_SEGMENT; i++){
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_a[ioffset], &h_a[ioffset], nBytes/N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(&d_b[ioffset], &h_b[ioffset], nBytes/N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
        sumArraysGPU<<<grid, block, 0, stream[i]>>>(&d_a[ioffset], &d_b[ioffset], &d_res[ioffset], iElem);
        CHECK(cudaMemcpyAsync(&h_res_from_gpu[ioffset], &d_res[ioffset], nBytes/N_SEGMENT, cudaMemcpyDeviceToHost, stream[i]));
        CHECK(cudaStreamAddCallback(stream[i],my_callback,(void *)(stream+i),0));
    }

    CHECK(cudaEventRecord(stop, 0));
     int counter=0;
    while (cudaEventQuery(stop)==cudaErrorNotReady)
    {
        counter++;
    }
    printf("cpu counter:%d\n",counter);

    t_stop = timeCount() - t_start;

    printf("Asynchronous Execution configuration<<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x, t_stop);
    checkResult(h_res,h_res_from_gpu,nElem);

    for(int i =0; i < N_SEGMENT; i++){
        CHECK(cudaStreamDestroy(stream[i]));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_res);
    cudaFree(h_res_from_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);





  
    return 0;
}