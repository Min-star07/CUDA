#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "../freshman.h"

#define TEMPLATE_SIZE 9
#define TEMP_RADIO_SIZE (TEMPLATE_SIZE/2)
#define BDIM 32

__constant__ int coef[TEMP_RADIO_SIZE];//if in midle of the program will be error

void convolution(int *in, int *out, int *template_, int size){
   for(int i = TEMP_RADIO_SIZE; i < size - TEMP_RADIO_SIZE; i++){
        for(int j = 1; j <=TEMP_RADIO_SIZE; j++){
            out[i] += template_[j - 1] * (in[i+j] - in[i-j]); 
        }
        //  printf("id : %d ;  CPU : %d\n",i,out[i]);
   }
}

__global__ void stencil_1d(int *in, int *out){
    // shared memory
    __shared__ int smem[BDIM + 2 *TEMP_RADIO_SIZE];

    // index of global memory
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;
    // printf("%d, %d\n", idx, in[idx]);
    // index to shared memory for stencil calculation
    int sidx = threadIdx.x + TEMP_RADIO_SIZE;

    // read data from gobal memory to shared memory
    smem[sidx] = in[idx];

    // read halo part to the shared memory
    if(threadIdx.x < TEMP_RADIO_SIZE){
        if(idx > TEMP_RADIO_SIZE)
        smem[sidx - TEMP_RADIO_SIZE] = in[idx -TEMP_RADIO_SIZE];
        if(idx < gridDim.x * blockDim.x - BDIM){
            smem[sidx + BDIM] = in[idx + BDIM];
        }
    }
    __syncthreads();

    if (idx<TEMP_RADIO_SIZE||idx>=gridDim.x*blockDim.x-TEMP_RADIO_SIZE)
        return;
    __syncthreads();
    int temp = 0;
    #pragma unroll
    for(int i = 1; i <= TEMP_RADIO_SIZE; i++){
        temp+=coef[i-1] * (smem[sidx + i] - smem[sidx-i]);
        // printf("temp : %d, %d\n" , coef[i-1], temp);
    }
    out[idx] = temp;
}


__global__ void stencil_1d_only_read(int *in, int *out, const int *__restrict__ dcoef){
    // shared memory
    __shared__ int smem[BDIM + 2 *TEMP_RADIO_SIZE];

    // index of global memory
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;
    // printf("%d, %d\n", idx, in[idx]);
    // index to shared memory for stencil calculation
    int sidx = threadIdx.x + TEMP_RADIO_SIZE;

    // read data from gobal memory to shared memory
    smem[sidx] = in[idx];

    // read halo part to the shared memory
    if(threadIdx.x < TEMP_RADIO_SIZE){
        if(idx > TEMP_RADIO_SIZE)
        smem[sidx - TEMP_RADIO_SIZE] = in[idx -TEMP_RADIO_SIZE];
        if(idx < gridDim.x * blockDim.x - BDIM){
            smem[sidx + BDIM] = in[idx + BDIM];
        }
    }
    __syncthreads();

    if (idx<TEMP_RADIO_SIZE||idx>=gridDim.x*blockDim.x-TEMP_RADIO_SIZE)
        return;
    __syncthreads();
    int temp = 0;
    #pragma unroll
    for(int i = 1; i <= TEMP_RADIO_SIZE; i++){
        temp+=dcoef[i-1] * (smem[sidx + i] - smem[sidx-i]);
        // printf("temp : %d, %d\n" , coef[i-1], temp);
    }
    out[idx] = temp;
}

int main(int argc, char** argv){

    int nElem = 1 << 5;
    if(argc > 1){
        nElem = atoi(argv[1]);
    }
    printf("Size : %d\n", nElem);
    int nByte = nElem * sizeof(int);
    int templ[]={-1,-2,2,1};
    int *h_a = (int*)malloc(nByte);
    int *h_b = (int*)malloc(nByte);
    int *h_b_gpu = (int*)malloc(nByte);
    InitialData(h_a, nElem);
   
    double t_start, t_stop;
    t_start = timeCount();
    convolution(h_a, h_b, templ, nElem);
    t_stop = timeCount() - t_start;
    printf("result in cpu run time is %.6f s\n" , t_stop);
    // for(int i =0; i <10; i++){
    //     printf("%d, %d\n", i, h_a[i]);
    // }
    int *d_a, *d_b;
    CHECK(cudaMalloc((void**)&d_a, nByte));
    CHECK(cudaMalloc((void**)&d_b, nByte));

    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b_gpu, nByte, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(coef, templ, TEMP_RADIO_SIZE *sizeof(int)));

    int dimx = 32;
   
    dim3 block(dimx);
    dim3 grid((nElem + block.x - 1)/block.x);
    t_start = timeCount();
    stencil_1d<<<grid, block>>>(d_a, d_b);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("G=>stencil_1d execution configuration <<<(%d, %d)>>> , run time is %.6f s\n" , grid.x,  block.x, t_stop);

    int *dcoef_ro;
    CHECK(cudaMalloc((void**)&dcoef_ro,TEMP_RADIO_SIZE * sizeof(int)));
    CHECK(cudaMemcpy(dcoef_ro,templ,TEMP_RADIO_SIZE * sizeof(int),cudaMemcpyHostToDevice));
    t_start = timeCount();
    stencil_1d_only_read<<<grid, block>>>(d_a, d_b, dcoef_ro);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("G=>stencil_1d_read_only execution configuration <<<(%d, %d)>>> , run time is %.6f s\n" , grid.x,  block.x, t_stop);
    

    
   return 0;
}