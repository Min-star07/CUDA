#include<cuda_runtime.h>
#include<stdio.h>
#include <omp.h>
#include "../freshman.h"
#define N 300000
__global__ void kernel_1()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_2()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_3()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_4()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}

int main(){
    // setenv("CUDA_DEVICE_MAX_CONNECTIONS","4",1);
    setGPU();
    int n_stream = 4;
    cudaStream_t *stream = (cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
    for(int i =0; i < n_stream; i++){
        cudaStreamCreate(&stream[i]);
    }
    dim3 block(1);
    dim3 grid(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    omp_set_num_threads(n_stream);
    cudaEventRecord(start, 0);
    #pragma omp parallel
    {      
        int i=omp_get_thread_num();
        kernel_1<<<grid,block,0,stream[i]>>>();
        kernel_2<<<grid,block,0,stream[i]>>>();
        kernel_3<<<grid,block,0,stream[i]>>>();
        kernel_4<<<grid,block,0,stream[i]>>>();
    }
    // dispatch job with breadth first way
    // for (int i = 0; i < n_streams; i++)
    // kernel_1<<<grid, block, 0, streams[i]>>>();
    // for (int i = 0; i < n_streams; i++)
    // kernel_2<<<grid, block, 0, streams[i]>>>();
    // for (int i = 0; i < n_streams; i++)
    // kernel_3<<<grid, block, 0, streams[i]>>>();
    // for (int i = 0; i < n_streams; i++)
    // kernel_4<<<grid, block, 0, streams[i]>>>();

    cudaEventRecord(stop, 0);

    // cudaDeviceSynchronize(stop);
    CHECK(cudaEventSynchronize(stop));

    float elapsed_time;

    // cudaEventElapsed(&elapsed_time, stop);
    cudaEventElapsedTime(&elapsed_time,start,stop);

    printf("elapsed time:%f ms\n",elapsed_time);

    for(int i =0; i < n_stream; i++){
        // cudaStreamDestory(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(stream);

    CHECK(cudaDeviceReset());

    return 0;
}