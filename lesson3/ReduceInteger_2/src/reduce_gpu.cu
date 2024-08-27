#include "reduce.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"
#include "timer.h"
__global__ void warmup(int *g_idata, int *g_odata, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    int *idata = g_idata + blockDim.x * blockIdx.x;
    // printf("===========warmup================\n");
    // printf("%d, %d, %d, %d, %d, %d, %d\n", tid, idx, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx]);
    if(idx > size) return;

    for(int stride = 1; stride < blockDim.x; stride*=2){
        if((tid % (2 * stride)) == 0){
        idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
    // printf("+++++++++++warmup+++++++++++++\n");
    // printf("%d, %d, %d, %d, %d, %d, %d\n", tid, idx, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx]);

}
__global__ void kernel_reduceNeighbored(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    int *idata = g_idata + blockDim.x * blockIdx.x;
    // printf("===========================\n");
    // printf("%d, %d, %d, %d, %d, %d, %d\n", tid, idx, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx]);
    if(idx > size) return;

    for(int stride = 1; stride < blockDim.x; stride*=2){
        if((tid % (2 * stride)) == 0){
        idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
    // printf("++++++++++++++++++++++++\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx]);

}

__global__ void kernel_reduceNeighboredLess(int *g_idata, int *g_odata, int size){;
}

__global__ void kernel_reduceInterleaved(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    int *idata = g_idata + blockDim.x * blockIdx.x ; //only lock memory

    // printf("===========================\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx], idata[5], g_idata[5]);
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x, blockDim.x, idata[idx], idata[tid],  idata[5], g_idata[5]);
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid], g_odata[idx]);
    if(idx > size) return;
    for(int stride = blockDim.x/2; stride >0; stride >>=1){
        if(tid < stride) {
    //         printf("++++=+++++++++++++===\n");
    //         printf("%d, %d, %d, %d, %d, %d, %d, %d\n", blockDim.x, blockIdx.x, tid, idx, stride, tid + stride, idata[tid], idata[tid + stride]);
    //         printf("++++=+++++++++++++===\n");
           idata[tid] += idata[tid + stride];
    //         printf("%d, %d, %d, %d, %d, %d, %d, %d\n", blockDim.x, blockIdx.x, tid, idx, stride, tid + stride, idata[tid], idata[tid + stride]);
        }
        __syncthreads();
    }

    if(tid == 0 ) g_odata[blockIdx.x] = idata[0];
}

__global__ void kernel_reduceUnroll2(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x*2;
    int tid = threadIdx.x;

    // float test =0.1;
    // printf("==========22222222222222222222222222=================\n");
    // printf("%d, %d, %d, %d, %d, %d\n", tid, idx,  blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid]);
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    // printf("------------222222222222222222222----------------\n");
    // printf("%d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,   g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
    if (tid >= size) return;

    if(idx + blockDim.x < size) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // printf("++++=++++++----222222222222-----+++++++===\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,  blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
    // printf("++++=++++++-----22222222222222----+++++++===\n");
    __syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
            // printf("++++++++++===\n");
		}
        // printf("%d, %d, %d, %d, %d, %d,  %d, %d\n", tid, idx, blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
        // printf("++++=+++++_________++++-+++++++===\n");
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void kernel_reduceUnroll4(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x*4;
    int tid = threadIdx.x;

    // printf("===============444444444444444444============\n");
    // printf("%d, %d, %d, %d, %d, %d\n", tid, idx,  blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid]);
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    // printf("---------------444444444444444444-------------\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,   g_idata[idx],g_idata[tid], idata[idx], idata[tid], idata[5]);
    if (tid >= size) return;

    if(idx + 3 * blockDim.x < size) {
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
    }
    __syncthreads();
    // printf("++++=++++++---------+++++++===\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,  blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
    // printf("++++=++++++---------+++++++===\n");
    __syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
            // printf("++++++++++===\n");
		}
        // printf("%d, %d, %d, %d, %d, %d,  %d, %d\n", tid, idx, blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
        // printf("++++=+++++_________++++-+++++++===\n");
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void kernel_reduceUnroll8(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x*8;
    int tid = threadIdx.x;

    // float test =0.1;
    // printf("===========================\n");
    // printf("%d, %d, %d, %d, %d, %d\n", tid, idx,  blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid]);
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // printf("----------------------------\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,   g_idata[idx],g_idata[tid], idata[idx], idata[tid], idata[5]);
    if (tid >= size) return;

    if(idx + 7 * blockDim.x < size) {
        g_idata[idx] += g_idata[idx + 1 * blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();
    // printf("++++=++++++---------+++++++===\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,  blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
    // printf("++++=++++++---------+++++++===\n");
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
            // printf("++++++++++===\n");
		}
        // printf("%d, %d, %d, %d, %d, %d,  %d, %d\n", tid, idx, blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
        // printf("++++=+++++_________++++-+++++++===\n");
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void kernel_reduceUnrollWarp8(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x*8;
    int tid = threadIdx.x;

    // float test =0.1;
    // printf("===========================\n");
    // printf("%d, %d, %d, %d, %d, %d\n", tid, idx,  blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid]);
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // printf("----------------------------\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,   g_idata[idx],g_idata[tid], idata[idx], idata[tid], idata[5]);
    if (tid >= size) return;

    // if(idx + 7 * blockDim.x < size) {
    //     g_idata[idx] += g_idata[idx + 1 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 2 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 3 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 4 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 5 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 6 * blockDim.x];
    //     g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    // }

    if(idx + 7 * blockDim.x < size){
        int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;
	}
    
    __syncthreads();
    // printf("++++=++++++---------+++++++===\n");
    // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", tid, idx, blockIdx.x,  blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
    // printf("++++=++++++---------+++++++===\n");

	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride > 32 ; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
            // printf("++++++++++===\n");
		}
        // printf("%d, %d, %d, %d, %d, %d,  %d, %d\n", tid, idx, blockIdx.x, blockDim.x, g_idata[idx],g_idata[tid], idata[idx], idata[tid]);
        // printf("++++=+++++_________++++-+++++++===\n");
		//synchronize within block
		__syncthreads();
	}
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid]+=vmem[tid+32];
        vmem[tid]+=vmem[tid+16];
        vmem[tid]+=vmem[tid+8];
        vmem[tid]+=vmem[tid+4];
        vmem[tid]+=vmem[tid+2];
        vmem[tid]+=vmem[tid+1];
    }
	// //write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void kernel_reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(tid >=size) return;

    if(idx + 7 * blockDim.x < size){
        g_idata[idx] += g_idata[idx + 1 * blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];

        // int a1=g_idata[idx];
		// int a2=g_idata[idx+blockDim.x];
		// int a3=g_idata[idx+2*blockDim.x];
		// int a4=g_idata[idx+3*blockDim.x];
		// int a5=g_idata[idx+4*blockDim.x];
		// int a6=g_idata[idx+5*blockDim.x];
		// int a7=g_idata[idx+6*blockDim.x];
		// int a8=g_idata[idx+7*blockDim.x];
		// g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;
    }
    __syncthreads();
    if(blockDim.x >= 1024 && tid < 512){
        idata[tid] += idata[tid + 512];
        // printf("============1111111111==========\n");
        // printf("%d %d %d %d %d\n", tid, idx, blockDim.x, blockIdx.x, gridDim.x );
    }
    __syncthreads();

    if(blockDim.x >= 512 && tid < 256){
        idata[tid] += idata[tid + 256];
        // printf("============22222222==========\n");
        // printf("%d %d %d %d %d\n", tid, idx, blockDim.x, blockIdx.x, gridDim.x );
    }
    __syncthreads();

    if(blockDim.x >= 256 && tid < 128){
        idata[tid] += idata[tid + 128];
        // printf("============333333333==========\n");
        // printf("%d %d %d %d %d\n", tid, idx, blockDim.x, blockIdx.x, gridDim.x );
    }
    __syncthreads();

    if(blockDim.x >= 128 && tid < 64){
        idata[tid] += idata[tid + 64];
        // printf("============444444444==========\n");
        // printf("%d %d %d %d %d\n", tid, idx, blockDim.x, blockIdx.x, gridDim.x );
    }
    __syncthreads();

    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid]+=vmem[tid+32];
        vmem[tid]+=vmem[tid+16];
        vmem[tid]+=vmem[tid+8];
        vmem[tid]+=vmem[tid+4];
        vmem[tid]+=vmem[tid+2];
        vmem[tid]+=vmem[tid+1];
    }
    __syncthreads();

    
    if(tid == 0);
    g_odata[blockIdx.x] = idata[0];

}
template <unsigned int iBlockSize>
__global__ void kernel_reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (tid >= size) return;

    if (idx + 7 * blockDim.x < size) {
        g_idata[idx] += g_idata[idx + 1 * blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    __syncthreads();

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}


void reduceNeighbored(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;

    timer.start_gpu();
    warmup<<<grid, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x);

    printf("Execution configuration <<<%d, %d>>> with reduceNeighbored (warmup): %d\n", grid.x, block.x, sum_gpu);

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    timer.start_gpu();
    kernel_reduceNeighbored<<<grid, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    sum_gpu = sumArraysOnGPU(h_odata, grid.x);
    printf("Execution configuration <<<%d, %d>>> with reduceNeighbored: %d\n", grid.x, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}

void reduceNeighboredLess(int *h_idata, int *h_odata, int nElem, int blockSize)
{
    printf("need further check===\n");
}
void reduceInterleaved(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    kernel_reduceInterleaved<<<grid, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x);
    printf("Execution configuration <<<%d, %d>>> with reduceInterleaved: %d\n", grid.x, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}


void reduceUnroll2(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    kernel_reduceUnroll2<<<grid.x/2, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/2);
    printf("Execution configuration <<<%d, %d>>> with unrolling2: %d\n", grid.x/2, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}


void reduceUnroll4(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;

    timer.start_gpu();
    kernel_reduceUnroll4<<<grid.x/4, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/4);
    printf("Execution configuration <<<%d, %d>>> with unrolling4: %d\n", grid.x/4, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}

void reduceUnroll8(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    kernel_reduceUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/8);
    printf("Execution configuration <<<%d, %d>>> with unrolling8: %d\n", grid.x/8, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}

void reduceUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize){
    int *d_idata, *d_odata;

    dim3 block(blockSize);
    dim3 grid((nElem + block.x-1)/block.x);
    
    CHECK(cudaMalloc((void**) &d_idata, nElem * sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    kernel_reduceUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, nElem);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/8);
    printf("Execution configuration <<<%d, %d>>> with unrolling8 warp: %d\n", grid.x/8, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}

void reduceCompleteUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize){
    int * d_idata, *d_odata;
    dim3 block(blockSize);
    dim3 grid((nElem + block.x -1)/ block.x);

    CHECK(cudaMalloc((void**) &d_idata, nElem *sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(g_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    // kernel_reduceUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, nElem);
    kernel_reduceCompleteUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, nElem);

    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/8);
    printf("Execution configuration <<<%d, %d>>> with completeunrolling 8 warp: %d\n", grid.x/8, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}

void reduceTemplateCompleteUnrollWarp8(int *h_idata, int *h_odata, int nElem, int blockSize){
    int * d_idata, *d_odata;
    dim3 block(blockSize);
    dim3 grid((nElem + block.x -1)/ block.x);

    CHECK(cudaMalloc((void**) &d_idata, nElem *sizeof(int)));
    CHECK(cudaMalloc((void**) &d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nElem * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(g_odata, h_odata, grid.x * sizeof(int), cudaMemcpyHostToDevice));

    Timer timer;
    timer.start_gpu();
    switch(blockSize)
	{
		case 1024:
			kernel_reduceCompleteUnrollWarp8 <1024><< <grid.x/8, block >> >(d_idata, d_odata, nElem);
		break;
		case 512:
			kernel_reduceCompleteUnrollWarp8 <512><< <grid.x/8, block >> >(d_idata, d_odata, nElem);
		break;
		case 256:
			kernel_reduceCompleteUnrollWarp8 <256><< <grid.x/8, block >> >(d_idata, d_odata, nElem);
		break;
		case 128:
			kernel_reduceCompleteUnrollWarp8 <128><< <grid.x/8, block >> >(d_idata, d_odata, nElem);
		break;
	}

    cudaDeviceSynchronize();
    timer.stop_gpu();
    timer.duration_cpu<Timer::ms>("reduce in GPU");

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    int sum_gpu = sumArraysOnGPU(h_odata, grid.x/8);
    printf("Execution configuration <<<%d, %d>>> with template completeunrolling 8 warp: %d\n", grid.x/8, block.x, sum_gpu);

    cudaFree(d_idata);
    cudaFree(d_odata);
}
