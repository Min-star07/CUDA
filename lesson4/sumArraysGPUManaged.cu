#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#define CHECK(call)                                                                \
{                                                                                  \
        const cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                                  \
        {                                                                          \
            printf("Error : %s : %d", __FILE__, __LINE__);                         \
            printf("code : %d,  reason : %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                               \
        }                                                                          \
}

void checResult(float *cpuRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for(int i = 0; i < N; i++){
        if(fabs(cpuRef[i] - gpuRef[i]) > epsilon){
            printf("Mismatch at index %d, CPU: %5.2f, GPU: %5.2f\n", i, cpuRef[i], gpuRef[i]);
            match = 0;
            break;
        }
    }
    if(match )
    printf("Results match!\n");
}

// void initializeData(float * ip, int size){
//     // generate different seed for random number
//     unsigned int seed = time(NULL);
//     for(int i = 0; i < size; i++){
//         ip[i] = rand_r(&seed) / ((float)RAND_MAX);
//     }
// }
void initializeData(float * ip, int size){
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for(int i = 0; i < size; i++){
        ip[i] = (float) (rand() &0xFF) / 10.0f;
    }
}

void sumArraysOnCPU(float * A, float *B, float *C, const int N){
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnGPU(float * A, float * B, float *C, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec *1.e-6);
}
int main(int argc, char ** argv){
    printf("%s Starting ....\n", argv[0]);
    //1. set up device
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Found %d CUDA device(s)\n", deviceCount);
    if(deviceCount <1){
        printf("No CUDA device found, exiting...\n");
        return 1;
    }
    else{
        printf("Found %d CUDA campatable GPU(s) in your computer.\n", deviceCount);
        for(int device = 0; device < deviceCount; device++){
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            printf("Device %d: %s has compute capability : Major: %d Minor: %d \n", device, deviceProp.name, deviceProp.major, deviceProp.minor);

            // set GPU
            CHECK(cudaSetDevice(device));
            printf("Set GPU %d.\n", device);
    }
    }
    //2. set up data size of vectors
    const int size = 1 << 24;
    size_t nBytes = size * sizeof(float);
    printf("Vectors of size %d.\n", size);
    //3. allocate memory on host and device
    float *A, *B, *GPURef, *CPURef;
    cudaMallocManaged((float **) & A, nBytes);
    cudaMallocManaged((float **) & B, nBytes);
    cudaMallocManaged((float **) & GPURef, nBytes);

    CPURef = (float *)malloc(nBytes);
    float iStart, iElaps;
    // 3.3 initialize data
    iStart = cpuSecond();
    int nx = 12;
    int ny = 6;
    int nxy = nx * ny;
    initializeData(A, nxy);
    initializeData(B, nxy);
    iElaps = cpuSecond() - iStart;
    printf("Initializing data on host took %f seconds.\n", iElaps);
    
   
    memset(CPURef, 0, nBytes);
    memset(GPURef, 0, nBytes);
    // initilaization data
   
    // 4. sum arrays on CPU
    iStart = cpuSecond();
    sumArraysOnCPU(A, B, CPURef, size);
     iElaps = cpuSecond() - iStart;
    printf("Time elapsed on CPU took %f seconds.\n", iElaps);
   
    // dim3 grid((size + block.x - 1) / block.x);
    dim3 block (256);
    dim3 grid((size + block.x - 1) / block.x);
    // 5.1 time to launch kernal
    iStart = cpuSecond();
    sumArrayOnGPU<<<grid, block>>>(A, B, GPURef, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution configration : <<<%d, %d>>> Time elapsed on GPU  %f seconds\n", grid.x, block.x, iElaps);
    // 6. copy data from device to host
  
    // 7. check results
    checResult(CPURef, GPURef, size);
    // 8. free memory on host and device
    cudaFree(A);
    cudaFree(B);
    free(CPURef);
    cudaFree(GPURef);
   
    printf("%s Done.\n", argv[0]);
    return 0;
}