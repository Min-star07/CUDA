#include <cuda_runtime.h>
#include <stdio.h>

#define cuda_check(msg)\
do{ \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess) \
    {\
        printf(stderr, "Fatal error : %s ( %s at %s : %d)\n", msg, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }\
}while(0)

void InitiData(float *array, int size){
    for(int i = 0; i < size; i++){
        array[i] = rand() / (float)RAND_MAX;
    }
}
const int DSIZE = 8;

__global__ void vadd1(float *A, float *B, float *C, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        C[idx] = A[idx] + B[idx];
         printf("%d , %d , %d, %d, %d, %.1f\n", idx, threadIdx.x, blockDim.x, blockIdx.x, gridDim.x, C[idx]);
    }
    __syncthreads();
}
// vector add kernel: C = A + B
__global__ void vadd2(const float *A, const float *B, float *C, int ds){
    int idx;
    for (idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds; idx += gridDim.x * blockDim.x) {// a grid-stride loop
        C[idx] = A[idx] + B[idx];                                                                  // do the vector (element) add here
    printf("%d , %d , %d, %d, %d, %.1f\n", idx, threadIdx.x, blockDim.x, blockIdx.x, gridDim.x, C[idx]);}
     __syncthreads();
}
int main(int argc, char** argv){
    float *h_a, *h_b, *h_c1, *h_c2;
    float *d_a, *d_b, *d_c1, *d_c2;

    size_t nBytes = DSIZE * sizeof(float);

    h_a = (float*)  malloc(nBytes);
    h_b = (float*)  malloc(nBytes);
    h_c1 = (float*)  malloc(nBytes);
    h_c2 = (float*)  malloc(nBytes);

    InitiData(h_a, DSIZE);
    InitiData(h_b, DSIZE);
    memset(h_c1, 0, DSIZE);
    memset(h_c2, 0, DSIZE);

    cudaMalloc((float**)&d_a, nBytes);
    cudaMalloc((float**)&d_b, nBytes);
    cudaMalloc((float**)&d_c1, nBytes);
    cudaMalloc((float**)&d_c2, nBytes);

    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    dim3 block(1);
    // dim3 grid(1);
    dim3 grid((DSIZE + block.x -1)/block.x);

    vadd1<<<grid, block>>>(d_a, d_b, d_c1, DSIZE);
    cudaMemcpy(h_c1, d_c1, nBytes, cudaMemcpyDeviceToHost);
    
    vadd2<<<grid, block>>>(d_a, d_b, d_c2, DSIZE);

    cudaMemcpy(h_c2, d_c2, nBytes, cudaMemcpyDeviceToHost);

    printf("finished vadd1 = %.1f, vadd2 = %.1f \n", h_c1[1], h_c2[1]);

    free(h_a);
    free(h_b);
    free(h_c1);
    free(h_c2);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c1);
    cudaFree(d_c2);

    return 0;

}