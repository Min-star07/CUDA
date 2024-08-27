
#ifndef FRESHMAN_H
#define FRESHMAN_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>


// CUDA ERROR CHECK
#define cudaCheckErrors(msg){\
    cudaError_t __err = cudaGetLastError();\
    if(__err != cudaSuccess){\
    fprintf(stderr, "Fatal error: %s ===> %s ===>  %s : %d\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
    fprintf(stderr, "FURTHER CHECK *****  ABORTION\n");\
    exit(1);\
};\
}

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

// Initilization data on host
void InitialData(int *vec, int size){
    
	for (int i = 0; i<size; i++)
	{
		vec[i] = i;
	}
}

// sum on host
void sum2ArraysOnCPU(float * a,float * b,float * res,const int size)
{

    for(int i=0;i<size;i++)
    {
        res[i]=a[i]+b[i];
    }

}

int  sum1ArraysOnCPU(int * a, const int size)
{
    int sum = 0;

    for(int i=0;i<size;i++)
    {
        sum +=a[i];
        // printf("%d, %d\n",i , sum);
    }
    
    return sum;
}

// compare the difference between CPU and GPU
void CheckResult(float *CPU_res, float *GPU_res, int size){
    float error_bar = 1e-6;
    for(int i =0; i < size; i++){
        if((CPU_res[i] - GPU_res[i] ) > error_bar){
            printf("ThE result is diffeence between GPU (%d, %.1f) and CPU (%d, %.1f)\n", i, GPU_res[i], i , CPU_res[i]);
            exit(1);
        }
    }
}



void setGPU(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount  < 1){
        printf("No CUDA device found, exiting...\n");
        exit(1);
    }
    else{
        for(int i = 0; i < deviceCount; i++ ){
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
             printf("Device %d: %s has compute capability : Major: %d Minor: %d \n", i, deviceProp.name, deviceProp.major, deviceProp.minor);

            // set GPU
            cudaSetDevice(i);
            printf("Set GPU %d.\n", i);
        }

    }


}


double timeCount()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

#endif//FRESHMAN_H