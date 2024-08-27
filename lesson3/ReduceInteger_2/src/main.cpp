#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include "utils.h"
#include "timer.h"
#include "reduce.h"

int seed;
int main(int argc, char **argv)
{
    Timer timer;
    int blockSize = 1024;
    int byteSize = 24;
    if (argc > 1)
    {
        blockSize = atoi(argv[1]);
        byteSize = atoi(argv[2]);
    }
    int nElem = 1 << byteSize;
    if (nElem > 1024 *1024){
        printf("data size : %.f MB\n", nElem * sizeof(int)/1024.0/1024.0);
    }
    else{
        printf("data size : %.f kb\n", nElem * sizeof(int) /1024.0);
    }

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);

    // 1.1 initialize the host memory
    int *h_idata = (int *)malloc(nElem * sizeof(int));
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(nElem * sizeof(int));

    // initialize Matrix
    seed = 1;
    initMatrix(h_idata, nElem, seed);
    memset(h_odata, 0, grid.x * sizeof(int));
    memcpy(tmp, h_idata, nElem * sizeof(int));
    // for (int i = 0; i < nElem; i ++){
    //     printf("%d, %d\n", i, h_idata[i]);
    // }
    printf("Initialization completed\n");
    
    // CPU Specifications
    timer.start_cpu();
    int sumCPU = reduceOnCPU(tmp, nElem);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("reduce in CPU");
    printf("Reduce on CPU: %d\n", sumCPU);
    // for (int i = 0; i < nElem; i ++){
    //     printf("%d, %d\n", i, h_idata[i]);
    // }
    // GPU reduceNeighbored
        reduceNeighbored(h_idata, h_odata, nElem, blockSize);

    // GPU reduceNeighbored need further check
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceNeighboredLess(h_idata, h_odata, nElem, blockSize);

    // GPU reduceInterleaved
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceInterleaved(h_idata, h_odata, nElem, blockSize);

    // GPU reduceUnroll
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceUnroll2(h_idata, h_odata, nElem, blockSize);

    // GPU reduceUnrol4
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceUnroll4(h_idata, h_odata, nElem, blockSize);
    
    // GPU reduceUnrol8
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceUnroll8(h_idata, h_odata, nElem, blockSize);

    // GPU reduceUnrol8 warp
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceUnrollWarp8(h_idata, h_odata, nElem, blockSize);
    
    // GPU reduceCompleteUnrolling8 
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceCompleteUnrollWarp8(h_idata, h_odata, nElem, blockSize);
    
    // GPU reduceCompleteUnrolling8 
        memset(h_odata, 0, grid.x * sizeof(int));
        reduceTemplateCompleteUnrollWarp8(h_idata, h_odata, nElem, blockSize);
}