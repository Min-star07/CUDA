#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

#define CUDA_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

#define BLOCKSIZE 16

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

inline static void __kernelCheck(const char* file, const int line) 
{
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(const char* format, ...) 
{
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}
void initMatrix(int *data, int size, int seed);
void printMatrix(int *data, int size);

void compareMat(int *h_data, int *d_data, int size);

#endif