#include <cuda_runtime.h>
#include<stdio.h>
int main(int argc, char **argv){
    printf("%s Starting \n", argv[0]);
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(error_id!= cudaSuccess){
        printf("cudaGetDeviceCount returned %d\n ->%s\n", error_id, cudaGetErrorString(error_id));
        return 1;
    }

    if(deviceCount==0){
        printf("No CUDA capable devices found.\n");
        return 1;
    }
    else{
        printf("%d CUDA capable devices detected.\n", deviceCount);
        
    }

    int dev = 0,  driverVersion = 0, runtimeVersion = 0;

    // for (int i = 0; i < deviceCount; i++){
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);   
        printf("CUDA Capability Major/Minor version :%d.%d\n", deviceProp.major, deviceProp.minor);
        // print the number of SM
        printf("Multiprocessor count : %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of global memory: %.2f MBytes (%llu bytes)\n", (float) deviceProp.totalGlobalMem/(pow(1024.0, 3)), (unsigned long long)deviceProp.totalGlobalMem);
        printf("GPU clock rate : %.0fMHz(%0.2fGHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6);
        printf("Memory Clock rate : %.0fMHz(%0.2fGHz)\n", deviceProp.memoryClockRate * 1e-3, deviceProp.memoryClockRate * 1e-6);
        printf("Memeory BUS Width : %.f-bit\n", (float)deviceProp.memoryBusWidth);
        if(deviceProp.l2CacheSize){
            printf("L2 Cache Size : %.0f KB\n", (float)deviceProp.l2CacheSize/(pow(1024.0, 2)));
        }
        printf("Max Texture Dimension Size (x y ,z)==> 1D =(%d), 2D = (%d, %d), 3D =(%d, %d, %d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("Max Layered Texture Size (dim )x layers ==> 1D=(%d) x %d, 2d=(%d,%d)x %d\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
        printf("Total constant memory : %lu bytes\n", deviceProp.totalConstMem);
        // printf share memory
        printf("Shared Memory per block : %lu bytes, %f kb\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/1024.0);
        printf("Total registers per block : %d\n", deviceProp.regsPerBlock);
        printf("Warp size : %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per block : %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum number of threads per multiprocessor : %d\n", deviceProp.maxThreadsPerMultiProcessor);
        // printf maximum size of each dimension of block
        printf("Maximum sizes of each dimension of a block : (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        // printf maximum size of each dimension of grid
        printf("Maximum sizes of each dimension of a grid : (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        // printf maximum memory pitch
        printf("Maximum memory pitch : %lu bytes\n", deviceProp.memPitch);
        // printf texture alignment
        printf("Texture alignment : %lu bytes\n", deviceProp.textureAlignment);
    // }
        exit(EXIT_SUCCESS);
}