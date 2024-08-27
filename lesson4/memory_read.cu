#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

// Function to calculate elapsed time
double timeCount(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

// Initialization of data on host
void InitData(float *vec, int size){
    for(int i = 0; i < size; i++){
        vec[i] = 1.0f;  // Simple initialization to 1.0 for easy validation
    }
}

// CUDA ERROR CHECK
#define cudaCheckErrors(msg){\
    cudaError_t __err = cudaGetLastError();\
    if(__err != cudaSuccess){\
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",\
            msg, cudaGetErrorString(__err), __FILE__, __LINE__);\
        exit(1);\
    }\
}

// CPU function to sum the vector

void sumArrays(float * a,float * b,float * res,int offset,const int size)
{

    for(int i=0,k=offset;k<size;i++,k++)
    {
        res[i]=a[k]+b[k];
        printf("%d, %d, %.1f, %.1f, %.1f, %.1f, %.1f\n", i, k, res[i], a[i], b[i], a[k], b[k]);
    }

}

void checkResult(float * hostRef,float * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

__global__ void vadd(float *a, float *b, float*res, int offset, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int k = idx + offset;
    if(k < size){
        res[idx] = a[k] + b[k];
        printf("%d, %d, %.1f, %.1f, %.1f, %.1f,  %.1f\n", idx, k, res[idx], a[idx], b[idx], a[k], b[k]);
    }
}


int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    int offset = 0;
    if(argc >1){
        offset = atoi(argv[1]);
    }

    const int nElem = 128;
    
    dim3 blockSize(32);
    dim3 gridSize((nElem + blockSize.x -1)/ blockSize.x);

    float *h_a, *h_b, *h_res;
    float *d_a, *d_b, *d_res, *res_from_gpu;

    size_t nBytes = nElem * sizeof(float);
    printf("vector size : %.3f MB\n", nBytes/1024.0/1024.0);

    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    h_res = (float*)malloc(nBytes);
    res_from_gpu = (float*)malloc(nBytes);

    InitData(h_a, nElem);
    InitData(h_b, nElem);
    memset(h_res, 0, nBytes);
    memset(res_from_gpu, 0, nBytes);

    cudaMalloc((void**)&d_a, nBytes);
    cudaMalloc((void**)&d_b, nBytes);
    cudaMalloc((void**)&d_res, nBytes);

    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, nBytes);
    // cudaMemcpy(d_res, h_res, nBytes, cudaMemcpyHostToDevice);
    
    double iStart,iElaps;
    iStart=timeCount();
    vadd<<<gridSize, blockSize>>>(d_a, d_b, d_res, offset, nElem);
    cudaDeviceSynchronize();
    iElaps=timeCount()-iStart;
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec --offset:%d \n",gridSize.x,blockSize.x,iElaps,offset);

    cudaMemcpy(res_from_gpu, d_res, nBytes, cudaMemcpyDeviceToHost);

    sumArrays(h_a,h_b, h_res,offset,nElem);

    checkResult(h_res,res_from_gpu,nElem);

    // printf("%.1f, %.1f, %.1f\n", h_a[0], h_b[0], h_res[0]);

    free(h_a);
    free(h_b);
    free(h_res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;

}