#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc, char ** argv){
    //define total data element
    int nElem = 1024;
    // define grid and block structure
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Grid.x = %d, block.x = %d\n", grid.x, block.x);
    // reset block
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid.x = %d, block.x = %d\n", grid.x, block.x);
    // reset block
    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid.x = %d, block.x = %d\n", grid.x, block.x);
    // reset block
    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid.x = %d, block.x = %d\n", grid.x, block.x);
    // reset device before you leave
    cudaDeviceReset();  // Make sure no CUDA-related memory leaks occur.
    return 0;

}