#include <cuda_runtime.h>
#include <stdio.h>
void checkresult(float *CPU_RESULT, float *GPU_RESULT, int size){
    float error = 1e-6;
    printf("%d" , size);
    for (int i = 0; i < size; i++)
    {
        float diff = CPU_RESULT[i] - GPU_RESULT[i];
        if(diff < -error || diff > error){
            printf("mismatch %d CPU_result : %.1f, GPU_result %.1f\n ", i, CPU_RESULT[i], GPU_RESULT[i]);
            exit(-1);
        }
        // else{
        //     printf("successful reulst \n");
        // }
    }
}
void addarrayonCPU(float *A, float *B, float *cpu_result, int size){
    for(int i = 0; i < size; i++){
        cpu_result[i] = A[i] + B[i];
    }
}

void initilization(float *before, float*after, int size){
    for(int i = 0; i < size; i++){
        before[i] = 1.0f;
        after[i] = 2.0f;
    }
}

__global__ void addarrayonGPU(float *a, float *b, float *c, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ;
    if(idx < size){
        c[idx] = a[idx] + b[idx];
        // printf("idx = %d, Result = %.1f\n", idx, c[idx]);
    }
}
int main(int argc, char **argv){
    // Set up device
    int dev = 0;
    cudaSetDevice(dev);

    // Getdeviceproperties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // check if support mapped memory
    if(!deviceProp.canMapHostMemory){
        printf("Device does not support mapped memory\n");
        return 1;
    }
    printf("Using Device %d : %s\n", dev, deviceProp.name);

    // set up vector size
    int ipower = 5;
    if(argc >1) ipower =atoi(argv[1]);
    int size = 1 << ipower;
    if(ipower <18)
        printf("memory size : %1.f kb\n", size * sizeof(float)/ 1024.0);
    else{
        printf("memory size : %1.f mb\n", size * sizeof(float) / (1024.0 * 1024.0));
    }

    // copy from host memory
    float *h_a, *h_b, *h_c, *CPURes;
    float *d_a, *d_b, *d_c;
    // int size = 1 << 24;
    size_t nbytes = size * sizeof(float);


    dim3 block(32);
    dim3 grid((size + block.x - 1) / block.x);

    // allocate the host memory void* malloc(size_t size);

    h_a = (float*) malloc(nbytes);
    h_b = (float*) malloc(nbytes);
    h_c = (float*) malloc(nbytes);
    CPURes = (float*) malloc(nbytes);

    // // initilizartion data
    // initilization(h_a, h_b, size);

    // allocate the device memory cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaMalloc((float **)&d_a, nbytes);
    cudaMalloc((float **)&d_b, nbytes);
    cudaMalloc((float **)&d_c, nbytes);

    // copy memory form host to device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nbytes, cudaMemcpyHostToDevice);

     // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    addarrayonGPU<<<grid, block>>>(d_a, d_b, d_c, size);
    // cudaDeviceSynchronize();
     // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the time taken by the kernel execution
    printf("Time taken by kernel: %f ms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);

    // check the result
    // checkresult(h_a, h_c, size)
    addarrayonCPU(h_a, h_b, CPURes, size);
    checkresult(CPURes, h_c, size);

    // free the memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // usinf zerocopy memory fro arrayA and B;
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((float **)&h_a, nbytes, flags);
    cudaHostAlloc((float **)&h_b, nbytes, flags); //
    cudaHostAlloc((float **)&h_c, nbytes, flags); //

    // initialize data at host side
    initilization(h_a, h_b, size);

    // copy the data from device to host
    // cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
    memset(CPURes, 0, nbytes);
    memset(h_c, 0, nbytes);

    // pass the pointer to device
    cudaHostGetDevicePointer((float **)&d_a, (void *)h_a, 0);
    cudaHostGetDevicePointer((float **)&d_b, (void *)h_b, 0);
    cudaHostGetDevicePointer((float **)&d_c, (void *)h_c, 0);

    // add on cpu
    addarrayonCPU(h_a, h_b, CPURes, size);
    
    // launch kernel
    // Create CUDA events
    // cudaEvent_t start_zerocopy, stop_zerocopy;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    addarrayonGPU<<<grid, block>>>(d_a, d_b, d_c, size);
    // cudaDeviceSynchronize();
     // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    // float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the time taken by the kernel execution
    printf("Time taken by kernel: %f ms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);

    // check the result
    checkresult(CPURes, h_c, size);

    // free the memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    free(CPURes);

    return 0;
}