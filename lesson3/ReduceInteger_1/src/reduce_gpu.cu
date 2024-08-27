#include "reduce.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"

__global__ void ReduceNeighboredWithDivergence(int *g_idata, int *g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if(idx >= size)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride< blockDim.x; stride *=2){
        if((tid %(2*stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}
__global__ void ReduceNeighboredWithoutDivergence(int *g_idata, int *g_odata, int size) { 
     // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if(idx >= size)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride< blockDim.x; stride *=2){
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index  + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void Reduceinterleaved(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if(idx >= size)
        return;
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride>>=1){
        // printf("%d %d %d %d \n", tid, idx, blockDim.x, stride);
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}
// unrooling
__global__ void ReduceUnrolling2(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    // unrolling 2 data blocks
    if(idx + blockDim.x < size) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride>>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        // synchronize with threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}
// unrooling
__global__ void ReduceUnrolling4(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    // unrolling 2 data blocks
    if(idx + 3 * blockDim.x < size) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 4; stride > 0; stride>>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        // synchronize with threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}

// unrooling
__global__ void ReduceUnrolling8(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling 2 data blocks
    if(idx + 7* blockDim.x < size)
    // {
    //       int a1 = g_idata[idx];
    //       int a2 = g_idata[idx + blockDim.x];
    //       int a3 = g_idata[idx + 2*blockDim.x];
    //       int a4 = g_idata[idx + 3*blockDim.x];
    //       int b1 = g_idata[idx + 4*blockDim.x];
    //       int b2 = g_idata[idx + 5*blockDim.x];
    //       int b3 = g_idata[idx + 6*blockDim.x];
    //       int b4 = g_idata[idx + 7*blockDim.x];
    //       g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    // }
     g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 8; stride > 0; stride>>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        // synchronize with threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}

// unrooling
__global__ void ReduceUnrollingWarps8(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling 2 data blocks
    if(idx + 7* blockDim.x < size)
    {
          int a1 = g_idata[idx];
          int a2 = g_idata[idx + blockDim.x];
          int a3 = g_idata[idx + 2*blockDim.x];
          int a4 = g_idata[idx + 3*blockDim.x];
          int b1 = g_idata[idx + 4*blockDim.x];
          int b2 = g_idata[idx + 5*blockDim.x];
          int b3 = g_idata[idx + 6*blockDim.x];
          int b4 = g_idata[idx + 7*blockDim.x];
          g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
   
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 8; stride > 32; stride>>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        // synchronize with threadblock
        __syncthreads();
    }
    // Warps unrolling
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}

// unrooling
__global__ void ReduceCompleteUnrollingWarps8(int *g_idata, int* g_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling 2 data blocks
    if(idx + 7* blockDim.x < size)
    {
          int a1 = g_idata[idx];
          int a2 = g_idata[idx + blockDim.x];
          int a3 = g_idata[idx + 2*blockDim.x];
          int a4 = g_idata[idx + 3*blockDim.x];
          int b1 = g_idata[idx + 4*blockDim.x];
          int b2 = g_idata[idx + 5*blockDim.x];
          int b3 = g_idata[idx + 6*blockDim.x];
          int b4 = g_idata[idx + 7*blockDim.x];
          g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
   
    __syncthreads();
    // in-place reduction in global memory and complete unroll
    if(blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads(); 
    if(blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
    // Warps unrolling
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // write result for this block to global mem
    if(tid ==0)
        g_odata[blockIdx.x] = idata[0];
}


void ReduceOnGPUWithDivergence(int *h_idata, int *h_odata, int size, int blockSize){
    // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceNeighboredWithDivergence<<<grid, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));

    
}
void ReduceOnGPUWithoutDivergence(int *h_idata, int *h_odata, int size, int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceNeighboredWithoutDivergence<<<grid, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}

void ReduceOnGPUwithinterleaved(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    Reduceinterleaved<<<grid, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}

void ReduceOnGPUwithunrolling2(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceUnrolling2<<<grid.x/2, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x /2  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}

void ReduceOnGPUwithunrolling4(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceUnrolling4<<<grid.x/4, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x /4  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}

void ReduceOnGPUwithunrolling8(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceUnrolling8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x /8  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
    // cudaDeviceReset();
}

void ReduceOnGPUwithunrollingWarps8(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceUnrolling8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x /8  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}


void ReduceOnGPUwithcompleteunrollingWarps8(int *h_idata, int *h_odata, int size,  int blockSize){
       // launch kernel
    dim3 block(blockSize);
    dim3 grid((size + block.x -1)/ block.x);
    // allocate device memeory
    int *g_idata, *g_odata;
    CUDA_CHECK(cudaMalloc((void**)&g_idata, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_odata, grid.x * sizeof(int)));
    // copy host data to device
    CUDA_CHECK(cudaMemcpy(g_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice));
    ReduceUnrolling8<<<grid.x/8, block>>>(g_idata, g_odata, size);
    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, g_odata, grid.x /8  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
   //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(g_odata));
    CUDA_CHECK(cudaFree(g_idata));
}