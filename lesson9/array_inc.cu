// This CUDA program allocates a large array in unified memory, 
// increments its elements using a GPU kernel, 
// and then checks if the operation succeeded. 

// Memory Prefetching: cudaMemPrefetchAsync is used to move memory between the CPU and GPU. 
// This improves memory access performance by ensuring the data resides in the appropriate memory space when accessed.
#include<cstdio>
#include<cstdlib>

// #define cudaCheckErrors(msg)\
// do{\
//     cudaError_t = cudaGetLastError();\
//     if(__err != cudaSuccess){\
//         fprintf(stderr, "Fatal error : %s (%s at %s : %d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
//         fprintf(stderr, "*****FAILED - ABORTING\n");\
//         exit(1);\
//     }\
// }while(0)

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes){
    cudaMallocManaged(&ptr, num_bytes);
}

__global__ void inc(int *array, size_t n){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < n){
        array[idx]++;
        idx += blockIdx.x * blockDim.x;
    } 
}
const size_t ds = 32;
// const size_t ds = 32ULL*1024ULL*1024ULL;

int main(){
    int *h_array;
    alloc_bytes(h_array, ds * sizeof(int));
    cudaCheckErrors("cudaMallocManaged Error");
    memset(h_array, 0, ds*sizeof(h_array[0]));
    cudaMemPrefetchAsync(h_array, ds*sizeof(int),0);
    inc<<<256, 256>>>(h_array, ds);
    cudaCheckErrors("kernel launch error");
    cudaMemPrefetchAsync(h_array, ds*sizeof(int[0]), cudaCpuDeviceId);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel execution error");
    for (int i = 0; i < ds; i++) 
        if (h_array[i] != 1) {printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1); return -1;}
    printf("success!\n"); 
    return 0;
}