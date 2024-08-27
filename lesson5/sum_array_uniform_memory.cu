#include<cuda_runtime.h>
#include<stdio.h>
#include"../freshman.h"

__global__ void sumUniformMemory(int *array, int *result, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int tid = threadIdx.x;

    int *idata = array + blockIdx.x * blockDim.x * 4;

    if(idx + 3 * blockDim.x < size){
        array[idx] += array[idx + 1 * blockDim.x];
        array[idx] += array[idx + 2 * blockDim.x];
        array[idx] += array[idx + 3 * blockDim.x];
    }

    if(idx > size) return;

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0) result[blockIdx.x] = idata[0];

}

int main(int argc, char** argv){
    setGPU();

    int *A,  *d_res;

    int nElem = 1 << 5;
    int nBytes = nElem * sizeof(int);
 
    CHECK(cudaMallocManaged((void**)&A, nBytes));
    CHECK(cudaMallocManaged((void**)&d_res, nBytes));


    InitialData(A, nElem);
    int sumOncpu = sum1ArraysOnCPU(A, nElem);
    
    CHECK(cudaMemset(d_res, 0 , nBytes));

    dim3 block(2);
    dim3 grid((nElem + block.x -1)/block.x);

    double iStart, iStop;
    iStart = timeCount();
    sumUniformMemory<<<grid.x/ 4,block>>>(A, d_res, nElem);
    cudaDeviceSynchronize();
    iStop = timeCount() - iStart;

    printf("Execution configurtion : <<<%d, %d>>>, time %.6f s\n", grid.x/4, block.x, iStop);
    
   
    int sumOngpu = sum1ArraysOnCPU(d_res, grid.x/4);
    
    if(sumOncpu - sumOngpu == 0){
        printf("consistence between GPU(%d) and CPU(%d)\n", sumOngpu, sumOncpu);
    }
    else{
        printf("GPU result : %d, CPU result %d \n", sumOngpu, sumOncpu);
    }

    cudaFreeHost(A);
    cudaFreeHost(d_res);
   

    return 0;
}