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
    // if(argc!= 2){
    //     std::cerr << "Usage:./reduce_cuda <block>" << std::endl;
    //     return 1;
    // }
    // initialization
    Timer timer;
    char str[100];
    int size = 1<<24;
    int blockSize = 512;
    if(argc >2){
        blockSize = atoi(argv[2]);
    }

    dim3 block(blockSize, 1);
    dim3 grid((size + blockSize - 1) / blockSize, 1);

    printf("grid %d block %d \n", grid.x, block.x);

    // 1.1 initialize the host memory
    int *h_idata = (int *)malloc(size * sizeof(int));
    int *h_odata = (int *)malloc(grid.x * sizeof(int));

    // initialize Matrix
    seed = 1;
    initMatrix(h_idata, size, seed);
    memset(h_odata, 0, grid.x * sizeof(int));
    printf("Initialization completed\n");

    // CPU Specifications
    timer.start_cpu();
    int sumCPU = ReduceOnCPU(h_idata, size);
    timer.stop_cpu();
    printf("Reduce on CPU: %d\n", sumCPU);
    timer.duration_cpu<Timer::ms>("reduce in CPU");

    // GPU warmmingup
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    printf("Reduce on GPU: %d\n", h_odata[0]);
    timer.duration_gpu("reduce in GPU (warmingup)");

    // GPU reduce in gpu with divergence
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithDivergence = 0;
    for (int i = 0; i <grid.x; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithDivergence += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with divergence, result:%d", sumOnGPUWithDivergence);
    timer.duration_gpu(str);

    // GPU reduce in gpu without divergence
    timer.start_gpu();
    ReduceOnGPUWithoutDivergence(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithoutDivergence = 0;
    for (int i = 0; i <grid.x; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithoutDivergence += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu without divergence, result:%d", sumOnGPUWithoutDivergence);
    timer.duration_gpu(str);

     // GPU reduce in gpu without divergence
    timer.start_gpu();
    ReduceOnGPUwithinterleaved(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithinterleaved = 0;
    for (int i = 0; i <grid.x; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithinterleaved += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with interleaverd, result:%d", sumOnGPUWithinterleaved);
    timer.duration_gpu(str);

    //  GPU reduce in gpu witn unrolling2
    timer.start_gpu();
    ReduceOnGPUwithunrolling2(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithunrolling2 = 0;
    for (int i = 0; i <grid.x/2; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithunrolling2 += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with unrolling2, result:%d", sumOnGPUWithunrolling2);
    timer.duration_gpu(str);

      // GPU reduce in gpu witn unrolling4
    timer.start_gpu();
    ReduceOnGPUwithunrolling4(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithunrolling4 = 0;
    for (int i = 0; i <grid.x/4; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithunrolling4 += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with unrolling4, result:%d", sumOnGPUWithunrolling4);
    timer.duration_gpu(str);

  // GPU reduce in gpu witn unrolling8
    timer.start_gpu();
    ReduceOnGPUwithunrolling8(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithunrolling8 = 0;
    for (int i = 0; i <grid.x/8; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithunrolling8 += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with unrolling8, result:%d", sumOnGPUWithunrolling8);
    timer.duration_gpu(str);

    // / GPU reduce in gpu witn unrolling8
    timer.start_gpu();
    ReduceOnGPUwithunrollingWarps8(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithunrollingWarps8 = 0;
    for (int i = 0; i <grid.x/8; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithunrollingWarps8 += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with unrolling warps 8, result:%d", sumOnGPUWithunrollingWarps8);
    timer.duration_gpu(str);

    // / GPU reduce in gpu witn unrolling8
    timer.start_gpu();
    ReduceOnGPUwithcompleteunrollingWarps8(h_idata, h_odata, size, blockSize);
    
    timer.stop_gpu();
    int sumOnGPUWithcompleteunrollingWarps8 = 0;
    for (int i = 0; i <grid.x/8; i++) 
    {
        // printf("%d, %d\n", i, h_odata[i]);
        sumOnGPUWithcompleteunrollingWarps8 += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with complete unrolling warps 8, result:%d", sumOnGPUWithcompleteunrollingWarps8);
    timer.duration_gpu(str);


  

}