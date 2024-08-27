#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <stdint.h>
// #include "timer.h"
void reduceNeighbored(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceNeighboredLess(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceInterleaved(int *h_idata, int *h_odata, int nElem, int blockSize);

void reduceUnroll2(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceUnroll4(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceUnroll8(int *h_idata, int *h_odata, int nElem, int blockSize);

void reduceUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceCompleteUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize);
void reduceTemplateCompleteUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize);

int sumArraysOnGPU(int *h_odata, int size);

extern int reduceOnCPU(int *data, int const nElem);

#endif /* __REDUCE_CPU_H__ */