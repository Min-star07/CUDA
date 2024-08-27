#include <cuda_runtime.h>
#include <stdio.h>
#include "../freshman.h"
#define DIM 1024

int recursiveReduce(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int* g_odata, int const size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if(tid > size) return;
    // convert global data pointer to the local 
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for(int stride = 1; stride < blockDim.x; stride *=2){
        if((tid % (2 *stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within block
		__syncthreads();
    }
   
    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduce_gmem(int *g_idata, int* g_odata, int const size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > size) return;
    //convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;
    if(blockDim.x >=1024 && tid < 512)idata[tid] += idata[tid + 512];
    __syncthreads();
    if(blockDim.x >=512 && tid < 256)idata[tid] += idata[tid + 256];
    __syncthreads();
    if(blockDim.x >=256 && tid < 128)idata[tid] += idata[tid + 128];
    __syncthreads();
    if(blockDim.x >=128 && tid < 64)idata[tid] += idata[tid + 64];
    __syncthreads();
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduce_smem(int *g_idata, int* g_odata, int const size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int smem[DIM];
    if (idx > size) return;
    //convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;
    // globalmemory to shared memory
    smem[tid] = idata[tid];
    // smem = g_idata + blockIdx.x*blockDim.x;
    __syncthreads();
    if(blockDim.x >=1024 && tid < 512)smem[tid] += smem[tid + 512];
    __syncthreads();
    if(blockDim.x >=512 && tid < 256)smem[tid] += smem[tid + 256];
    __syncthreads();
    if(blockDim.x >=256 && tid < 128)smem[tid] += smem[tid + 128];
    __syncthreads();
    if(blockDim.x >=128 && tid < 64)smem[tid] += smem[tid + 64];
    __syncthreads();
    if(tid < 32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = smem[0];
}

__inline__ __device__ int warpReduce(int localSum){
    // printf("==================--------------\n");
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 16);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 8);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 4);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 2);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 1);
    // printf("%d \n", localSum);
}
__global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int size){
    // set threadID
    __shared__ int smem[DIM];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 1
    int mySum = g_idata[idx];
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;
    // printf("%d %d %d\n", idx, mySum, g_idata[idx]);
    // 2
    // printf("================\n");
    mySum = warpReduce(mySum);
    // printf("%d %d %d %d %d\n", idx, laneIdx, warpIdx, mySum, g_idata[idx]);
    // 3
    if(laneIdx ==0)
    smem[warpIdx] = mySum;
    __syncthreads();
    // 4
    mySum = (threadIdx.x < DIM) ?smem[laneIdx]:0;
    if(warpIdx==0){
        mySum=warpReduce(mySum);
    }
    // 5
    if(threadIdx.x ==0)
    g_odata[blockIdx.x] = mySum;
}

int main(int argc, char**argv){
    setGPU();

    // bool bResult = false;

    int nElem =1 << 12;
    printf("with array size %d \n",nElem);

    // execution configuration
    int blocksize = 1024;
    if(argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x -1)/block.x, 1);
    printf("execution configuration : <<<%d, %d>>>\n", grid.x, block.x);

    // allocate the host memory
    size_t nBytes = nElem * sizeof(int);
    int *h_a = (int*)malloc(nBytes); 
    int *h_res_from_cpu = (int*)malloc(nBytes); 
    int *h_res_from_gpu = (int*)malloc(grid.x * sizeof(int));

    // initialize the data
    InitialData(h_a, nElem);
    memset(h_res_from_cpu, 0, nBytes);
    memset(h_res_from_gpu, 0, grid.x * sizeof(int));

    // allocate the device memory
    int *d_a, *d_res;
    CHECK(cudaMalloc((void**)&d_a, nBytes));
    CHECK(cudaMalloc((void**)&d_res, grid.x * sizeof(int)));
    
    double t_start, t_stop;

    // cpu reduction
    memcpy(h_res_from_cpu, h_a, nBytes);
    t_start = timeCount();
    int sum_cpu = recursiveReduce(h_res_from_cpu, nElem);
    t_stop = timeCount() - t_start;
    printf("cpu reduce : elapsed %lf ms cpu_sum: %d\n", t_stop, sum_cpu);

    // warmup
    // for(int i =0; i < nElem; i++){
    //     printf("%d, %d\n", i, h_a[i]);
    // }
    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, grid.x * sizeof(int)));
    t_start = timeCount();
    warmup<<<grid, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int sum_gpu = 0;
    for(int i =0; i < grid.x; i++){
        sum_gpu += h_res_from_gpu[i];
        // printf("%d, %d, %d\n", i, h_res_from_gpu[i], sum_gpu);
    }
    printf("gpu warmup: elapsed %lf s gpu_sum: %d\n", t_stop, sum_gpu);

    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, grid.x * sizeof(int)));
    t_start = timeCount();
    reduce_gmem<<<grid, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i =0; i < grid.x; i++){
        sum_gpu += h_res_from_gpu[i];
        // printf("%d, %d, %d\n", i, h_res_from_gpu[i], sum_gpu);
    }
    printf("gpu reduce_gmem: elapsed %lf s gpu_sum: %d\n", t_stop, sum_gpu);

    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, grid.x * sizeof(int)));
    t_start = timeCount();
    reduce_smem<<<grid, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i =0; i < grid.x; i++){
        sum_gpu += h_res_from_gpu[i];
        // printf("%d, %d, %d\n", i, h_res_from_gpu[i], sum_gpu);
    }
    printf("gpu reduce_smem: elapsed %lf s gpu_sum: %d\n", t_stop, sum_gpu);

    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, grid.x * sizeof(int)));
    t_start = timeCount();
    reduceShfl<<<grid, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i =0; i < grid.x; i++){
        sum_gpu += h_res_from_gpu[i];
        // printf("%d, %d, %d\n", i, h_res_from_gpu[i], sum_gpu);
    }
    printf("gpu reduceShfl: elapsed %lf s gpu_sum: %d\n", t_stop, sum_gpu);


    return 0;
}