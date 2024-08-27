#include<cuda_runtime.h>
#include<stdio.h>
#include "../freshman.h"
#define BDIM 16
#define SEGM 4
// __global__ void test_shfl_broadcast(int *in, int *out, const int srcLane){
//     int value = in[threadIdx.x];
//     value = __shfl(value, srcLane, BDIM);
// }

__global__ void test_shfl_broadcast(int *in, int *out, const int srcLane) {
    int value = in[threadIdx.x];
    // Perform the shuffle broadcast
    value = __shfl_sync(value, srcLane, BDIM);
    // Store the result in the output array
    out[threadIdx.x] = value;
}

// __global__ void test_shfl_up(int *in, int *out, const int delta){
//     int value = in[threadIdx.x];
//     value = __shfl_up_sync(value, delta, BDIM);
//     out[threadIdx.x] = value;
// }

__global__ void test_shfl_up(int *in, int *out, const int delta) {
    int value = in[threadIdx.x];
    // Perform the shuffle up operation
    value = __shfl_up_sync(0xFFFFFFFF, value, delta, BDIM);
    // Store the result in the output array
    out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int *in, int *out, const int delta) {
    int value = in[threadIdx.x];
    // Perform the shuffle up operation
    value = __shfl_down_sync(0xFFFFFFFF, value, delta, BDIM);
    // Store the result in the output array
    out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap(int *in, int *out, const int offset) {
    int value = in[threadIdx.x];
    // Perform the shuffle up operation
    value = __shfl_sync(0xFFFFFFFF, value, threadIdx.x + offset, BDIM);
    // Store the result in the output array
    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int *in, int *out, const int mask) {
    int value = in[threadIdx.x];
    // Perform the shuffle up operation
    value = __shfl_xor_sync(0xFFFFFFFF, value, mask, BDIM);
    // Store the result in the output array
    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int *in, int *out, const int mask) {
    int idx = threadIdx.x * SEGM;
    
    int value[SEGM];
    // printf("%d %d\n",  idx,  in[idx]);
    // Perform the shuffle up operation
    for(int i =0; i < SEGM; i++)
        value[i] = in[idx +i];
    // printf("%d %d %d %d %d %d\n", i, idx, idx+i, value[i], in[i], in[idx+i]);

    value[0] = __shfl_xor_sync(0xFFFFFFFF, value[0], mask, BDIM);
    value[1] = __shfl_xor_sync(0xFFFFFFFF, value[1], mask, BDIM);
    value[2] = __shfl_xor_sync(0xFFFFFFFF, value[2], mask, BDIM);
    value[3] = __shfl_xor_sync(0xFFFFFFFF, value[3], mask, BDIM);
    // printf("=====================\n");
    // printf("%d %d %d\n",  idx, value[idx], in[idx]);
    // // Store the result in the output array
    for(int i =0; i < SEGM; i++)
    out[idx + i] = value[i];
}

__inline__ __device__
void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx){
   bool pred=((laneIdx%(2))==0);
    if(pred) {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
    value[secondIdx] = __shfl_xor_sync(0xFFFFFFFF, value[secondIdx], mask, BDIM);
    if(pred){
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

}
__global__ void test_shfl_swap(int *in, int* out, int const mask, int firstIdx, int secondIdx){
    // 1. 
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for(int i =0; i < SEGM; i++){
        value[i] = in[idx + i];
    }
    // 2.
    swap(value, threadIdx.x, mask, firstIdx, secondIdx);
    
    // 3
    for(int i = 0; i < SEGM; i++){
        out[idx + i] = value[i];
    }
}
int main(int argc, char** argv){
    setGPU();

    // int dimx = BDIM;
    unsigned int nElem = BDIM;
    int nBytes = nElem * sizeof(int);
    int kernel_num = 0;

    if(argc >= 2){
        kernel_num = atoi(argv[1]);
    }

    int h_a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int *h_res_from_gpu = (int*)malloc(nBytes);

    int *d_a = NULL, *d_res = NULL;
    CHECK(cudaMalloc((void**)&d_a, nBytes));
    CHECK(cudaMalloc((void**)&d_res, nBytes));
    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, nBytes));

    dim3 block(32);
    dim3 grid((nElem + block.x -1)/ block.x);

    switch(kernel_num){
        case 0:
            test_shfl_broadcast<<<1, BDIM>>>(d_a, d_res, 2);
            break;
        case 1:
            test_shfl_up<<<1, BDIM>>>(d_a, d_res, 2);
            break;
        case 2:
            test_shfl_down<<<1, BDIM>>>(d_a, d_res, 2);
            break;
        case 3:
            test_shfl_wrap<<<1, BDIM>>>(d_a, d_res, 2);
            break;
        case 4:
            test_shfl_xor<<<1, BDIM>>>(d_a, d_res, 1);
            break;
        case 5:
            test_shfl_xor_array<<<1, BDIM/4>>>(d_a, d_res, 1);
            break;
        case 6:
            test_shfl_swap<<<1, BDIM/4>>>(d_a, d_res, 1, 0, 3);
            break;
    }

    CHECK(cudaMemcpy(h_res_from_gpu, d_res, nBytes,cudaMemcpyDeviceToHost));
    //show result
    printf("input:\t");
    for(int i=0;i<nElem;i++)
        printf("%4d ",h_a[i]);
    printf("\noutput:\t");
    for(int i=0;i<nElem;i++)
        printf("%4d ",h_res_from_gpu[i]);
    printf("\n");
    CHECK(cudaMemset(d_res,0,nBytes));
    // stencil 1d read only


    cudaFree(d_a);
    cudaFree(d_res);
    free(h_res_from_gpu);
    // free(h_a);
    cudaDeviceReset();
    return 0;
}