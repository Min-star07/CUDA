#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv){
    // set ip device
    int dev = 0;
    cudaSetDevice(dev);

    // memory size
    unsigned int size = 1 << 22;
    unsigned int nbytes = size * sizeof(float);

    // Get device igormation
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device : %d : %s => Memory : size : %d, nbytes : %5.2f MB\n", dev, prop.name, size, nbytes / (1024.0 * 1024.0));

    // allocate the host memory
    float *h_a;
    h_a = (float *)malloc(nbytes);
    // initialize the host memory
    for (int i = 0; i < size; i ++){
        h_a[i] = 0.5f;
    }

    // allocate the device memeory
    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);
    // copy the host memory to the device memory
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    // copy the device memeory to the host
    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    // free the device memory
    cudaFree(d_a);
    // free the host memory
    free(h_a);
    cudaDeviceReset();
    return 0;
}