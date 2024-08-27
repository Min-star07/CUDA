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

void  initializeData(float *ip, int size){
    for (int i = 0; i < size; i++){
        ip[i] = i;
    }
}

void sumMatrixOnCPU(float * A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for(int i = 0; i < ny; i++){
        for(int j = 0; j < nx; j++){
            ic[j] = ia[j] + ib[j];
        }
        ia += nx, ib += nx, ic += nx;
    }
}

__global__ void sumMatrixOnGPU(float * A, float * B, float *C, const int nx, const int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if(ix < nx && iy < ny ){
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
    const int nx = 1 << 14;
    const int ny = 1 << 14;
    const int size = nx * ny;
    printf("Vectors of size %d.\n", size);
    //3. allocate memory on host and device
    float *h_A, *h_B;
    float *d_A, *d_B, *d_C;
    float *CPURef, *GPURef;
    // 3.1 allocate memory on host
    size_t nBytes = size * sizeof(float);
    printf("Maxtrix size : nx = %d, ny = %d \n", nx, ny);
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    CPURef = (float *)malloc(nBytes);
    GPURef = (float *)malloc(nBytes);
    double iStart, iElaps;
    // 3.2 allocate memory on device
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));
    // 3.3 initialize data
    iStart = cpuSecond();
    initializeData(h_A, size);
    initializeData(h_B, size);
    iElaps = cpuSecond() - iStart;
    printf("Initializing data on host took %f seconds.\n", iElaps);
    memset(CPURef, 0, nBytes);
    memset(GPURef, 0, nBytes);
    // 4. sum arrays on CPU
    iStart = cpuSecond();
    sumMatrixOnCPU(h_A, h_B, CPURef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("Time elapsed on CPU took %f seconds.\n", iElaps);
    // 4. copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    // 5. launch kernal
    // dim3 block (size);
    // dim3 grid((size + block.x - 1) / block.x);
    int dimx = 256;
    // int dimy = 1;
    dim3 block(dimx);
    dim3 grid((nx + block.x - 1) / block.x, ny);
    // 5.1 time to launch kernal
    iStart = cpuSecond();
    sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution configration : <<<(%d, %d), (%d,%d)>>> Time elapsed on GPU  %f seconds\n", grid.x, grid.y,  block.x, block.y, iElaps);
    // 6. copy data from device to host
    CHECK(cudaMemcpy(GPURef, d_C, nBytes, cudaMemcpyDeviceToHost));
    // 7. check results
    checResult(CPURef, GPURef, size);
    // 8. free memory on host and device
    free(h_A);
    free(h_B);
    free(CPURef);
    free(GPURef);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    printf("%s Done.\n", argv[0]);
    return 0;
}