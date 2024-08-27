#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printhelloworld(int const iSize, int Depth) {
    int tid = threadIdx.x;
    printf("Hello World! %d, Depth: %d\n", tid, Depth);

    if (iSize == 1)
        return;

    // Calculate the number of threads for the next iteration
    int nthreads = iSize >> 1;

    // Recursive kernel launch
    if (tid == 0 && nthreads > 0) {
        printhelloworld<<<1, nthreads>>>(nthreads, Depth + 1);
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    printhelloworld<<<1, 8>>>(8, 0);
    cudaDeviceSynchronize();
    return 0;
}
