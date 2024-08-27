#include <stdio.h>
__global__ void helloFromGPU(){
    if(threadIdx.x == 5)
        printf("Hello world from GPU thred 5!\n");
}
int main(){
    printf("Hello world from CPU!\n");
    helloFromGPU<<<1,10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}