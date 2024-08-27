#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <stdint.h>
void ReduceOnGPUWithDivergence(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUWithoutDivergence(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithinterleaved(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithunrolling2(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithunrolling4(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithunrolling8(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithunrollingWarps8(int *h_idata, int *h_odata, int size, int blockSize);
void ReduceOnGPUwithcompleteunrollingWarps8(int *h_idata, int *h_odata, int size, int blockSize);

extern int ReduceOnCPU(int *data, int const size);
#endif /* __REDUCE_CPU_H__ */