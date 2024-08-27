#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "../freshman.h"
#define DIM 1024

float get_ibd(int w, int r, float t){
    printf("%d, %d, %f\n", w, r, t);
    float bd = (w + r) * sizeof(int)/1.0e9/t;
    return bd;
}
int reduceinterger(int * array, int size){
    if(size == 1) return array[0];
    const int stride = size / 2;
    if(size %2 == 1){
        for(int i = 0; i < stride ; i++){
            array[i] += array[i + stride]; 
        }
        array[0] += array[size-1];
    }else{
        for(int i = 0; i < stride ; i++){
        array[i] += array[i + stride]; 
    }
    }
   
    return reduceinterger(array, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    // printf("%d, %d,%d\n", tid, idx,g_idata[idx]);

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx > size) return;

    for(int stride = blockDim.x / 2; stride > 0; stride >>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
         __syncthreads();
        // printf("%d, %d, %d\n", tid, idx, idata[tid]);
    }
    

    if(tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    // printf("%d,  %d\n", blockIdx.x, idata[0]);
    }
}

__global__ void reduce_interger_unroll4(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if(idx > size) return;

    if(idx + 3 * blockDim.x < size){
        g_idata[idx] += g_idata[idx + 1 * blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
    }
     __syncthreads();

    for(int stride = blockDim.x / 2; stride > 32; stride >>=1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
         __syncthreads();
    }
    if(tid < 32)
    {
            volatile int *vmem = idata;
            vmem[tid]+=vmem[tid+32];
            vmem[tid]+=vmem[tid+16];
            vmem[tid]+=vmem[tid+8];
            vmem[tid]+=vmem[tid+4];
            vmem[tid]+=vmem[tid+2];
            vmem[tid]+=vmem[tid+1];
    }
     __syncthreads();
    if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduce_interger_complete_unroll4(int *g_idata, int *g_odata, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if(idx > size) return;

    if(idx + 3 * blockDim.x < size){
        g_idata[idx] += g_idata[idx + 1 * blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
    }
     __syncthreads();

   if(blockDim.x >=1024 && tid< 512){
        idata[tid] += idata[tid+512];
   }
   __syncthreads();
   if(blockDim.x >=512 && tid< 256){
        idata[tid] += idata[tid+256];
   }
   __syncthreads();
   if(blockDim.x >=256 && tid< 128){
        idata[tid] += idata[tid+128];
   }
   __syncthreads();
   if(blockDim.x >=128 && tid < 64){
        idata[tid] += idata[tid+64];
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
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduce_interger_share(int *g_idata, int *g_odata, int size){


        __shared__ int tile[DIM];

        // int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int tid = threadIdx.x;

        int *idata = g_idata + blockIdx.x * blockDim.x;
        tile[tid] = idata[tid];
        __syncthreads();

        if(tid > size) return;

        for(int stride = blockDim.x / 2; stride > 32; stride >>=1){
            if(tid < stride){
                tile[tid] += tile[tid + stride];
            }
            __syncthreads();
        }
        if(tid < 32)
        {
            volatile int *vmem = tile;
            vmem[tid]+=vmem[tid+32];
            vmem[tid]+=vmem[tid+16];
            vmem[tid]+=vmem[tid+8];
            vmem[tid]+=vmem[tid+4];
            vmem[tid]+=vmem[tid+2];
            vmem[tid]+=vmem[tid+1];
        }
        __syncthreads();
        if (tid == 0)
            g_odata[blockIdx.x] = tile[0];
}

__global__ void reduce_interger_unroll4_share(int *g_idata, int *g_odata, int size){
        __shared__ int tile[DIM];
        int idx = threadIdx.x + blockIdx.x * blockDim.x *4;
        int tid = threadIdx.x;
        
        if(idx > size) return;

        int tmpsum = 0;
        if(idx + 3 * blockDim.x < size){
            int a1 = g_idata[idx];
            int a2 = g_idata[idx + 1 * blockDim.x];
            int a3 = g_idata[idx + 2 * blockDim.x];
            int a4 = g_idata[idx + 3 * blockDim.x];
            tmpsum = a1 + a2 + a3 + a4;
        }
        
        tile[tid] = tmpsum; 
         __syncthreads();

        for(int stride = blockDim.x / 2; stride > 32; stride >>=1){
            if(tid < stride){
                tile[tid] += tile[tid + stride];
            }
            __syncthreads();
        }
         if(tid < 32)
        {
            volatile int *vmem = tile;
            vmem[tid]+=vmem[tid+32];
            vmem[tid]+=vmem[tid+16];
            vmem[tid]+=vmem[tid+8];
            vmem[tid]+=vmem[tid+4];
            vmem[tid]+=vmem[tid+2];
            vmem[tid]+=vmem[tid+1];
        }
        __syncthreads();
        if (tid == 0)
            g_odata[blockIdx.x] = tile[0];
}
// __global__ void reduce_interger_unroll4_share(int *g_idata, int *g_odata, int size) {
//     __shared__ int tile[DIM];
//     int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
//     int tid = threadIdx.x;

//     if (idx >= size) return;

//     int tmpsum = 0;

//     if (idx + 3 * blockDim.x < size) {
//         int a1 = g_idata[idx];
//         int a2 = g_idata[idx + blockDim.x];
//         int a3 = g_idata[idx + 2 * blockDim.x];
//         int a4 = g_idata[idx + 3 * blockDim.x];
//         tmpsum = a1 + a2 + a3 + a4;
//     }

//     tile[tid] = tmpsum;
//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
//         if (tid < stride) {
//             tile[tid] += tile[tid + stride];
//         }
//         __syncthreads();
//     }

//     if (tid < 32) {
//         volatile int *vmem = tile;
//         vmem[tid] += vmem[tid + 32];
//         vmem[tid] += vmem[tid + 16];
//         vmem[tid] += vmem[tid + 8];
//         vmem[tid] += vmem[tid + 4];
//         vmem[tid] += vmem[tid + 2];
//         vmem[tid] += vmem[tid + 1];
//     }

//     __syncthreads();

//     if (tid == 0) {
//         g_odata[blockIdx.x] = tile[0];
//     }
// }


__global__ void reduce_interger_complete_unroll4_share(int *g_idata, int *g_odata, int size){
        __shared__ int tile[DIM];

        int idx = threadIdx.x + blockIdx.x * blockDim.x *4;
        int tid = threadIdx.x;
        
        if(tid > size) return;

        int tmpsum = 0;
        if(idx + 3 * blockDim.x <= size){
            int a1 = g_idata[idx];
            int a2 = g_idata[idx + 1 * blockDim.x];
            int a3 = g_idata[idx + 2 * blockDim.x];
            int a4 = g_idata[idx + 3 * blockDim.x];
            tmpsum = a1 + a2 + a3 + a4;
        }
        
        tile[tid] = tmpsum; 
        __syncthreads();

        if(blockDim.x >=1024 && tid< 512){
                tile[tid] += tile[tid+512];
        }
        __syncthreads();
        if(blockDim.x >=512 && tid< 256){
                tile[tid] += tile[tid+256];
        }
        __syncthreads();
        if(blockDim.x >=256 && tid< 128){
                tile[tid] += tile[tid+128];
        }
        __syncthreads();
        if(blockDim.x >=128 && tid < 64){
                tile[tid] += tile[tid+64];
        }
        __syncthreads();
        if(tid < 32)
        {
            volatile int *vmem = tile;
            vmem[tid]+=vmem[tid+32];
            vmem[tid]+=vmem[tid+16];
            vmem[tid]+=vmem[tid+8];
            vmem[tid]+=vmem[tid+4];
            vmem[tid]+=vmem[tid+2];
            vmem[tid]+=vmem[tid+1];
        }
        __syncthreads();
        if (tid == 0)
            g_odata[blockIdx.x] = tile[0];
}

__global__ void reduce_interger_complete_unroll4_dynshare(int *g_idata, int *g_odata, int size){
        extern __shared__ int tile[];

        int idx = threadIdx.x + blockIdx.x * blockDim.x *4;
        int tid = threadIdx.x;
        
        if(tid > size) return;

        int tmpsum = 0;
        if(idx + 3 * blockDim.x <= size){
            int a1 = g_idata[idx];
            int a2 = g_idata[idx + 1 * blockDim.x];
            int a3 = g_idata[idx + 2 * blockDim.x];
            int a4 = g_idata[idx + 3 * blockDim.x];
            tmpsum = a1 + a2 + a3 + a4;
        }
        
        tile[tid] = tmpsum; 
        __syncthreads();

        if(blockDim.x >=1024 && tid< 512){
                tile[tid] += tile[tid+512];
        }
        __syncthreads();
        if(blockDim.x >=512 && tid< 256){
                tile[tid] += tile[tid+256];
        }
        __syncthreads();
        if(blockDim.x >=256 && tid< 128){
                tile[tid] += tile[tid+128];
        }
        __syncthreads();
        if(blockDim.x >=128 && tid < 64){
                tile[tid] += tile[tid+64];
        }
        __syncthreads();
        if(tid < 32)
        {
            volatile int *vmem = tile;
            vmem[tid]+=vmem[tid+32];
            vmem[tid]+=vmem[tid+16];
            vmem[tid]+=vmem[tid+8];
            vmem[tid]+=vmem[tid+4];
            vmem[tid]+=vmem[tid+2];
            vmem[tid]+=vmem[tid+1];
        }
        __syncthreads();
        if (tid == 0)
            g_odata[blockIdx.x] = tile[0];
}
int main(int argc, char** argv){
    setGPU();

    int nElem = 1 << 12;
    printf("vec size : %d\n", nElem);
    int nByte = nElem * sizeof(nElem);

    if(argc  >1){
        nElem = atoi(argv[1]);
    }

    int *h_a, *h_res_from_gpu;
    int *d_a, *d_res;

    dim3 block(1024);
    dim3 grid((nElem + block.x -1) / block.x);

    h_a = (int*)malloc(nByte);
    h_res_from_gpu = (int*)malloc(grid.x * sizeof(int));

    InitialData(h_a, nElem);
    memset(h_res_from_gpu, 0, grid.x * sizeof(int));
    double t_start, t_stop;
    // result from cpu
    t_start = timeCount();
    int sum_cpu = reduceinterger(h_a, nElem);
    t_stop = timeCount() - t_start;
    printf("result from cpu %d, run time is %.6f s\n" , sum_cpu, t_stop);

    // for(int i = 0; i < nElem; i ++){
    //     printf("%d %d\n", i, h_a[i]);
    // }
    InitialData(h_a, nElem);
    CHECK(cudaMalloc((void**)&d_a, nByte));
    CHECK(cudaMalloc((void**)&d_res, grid.x * sizeof(int)));
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_res, h_res_from_gpu, grid.x * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0 ,grid.x * sizeof(int)));

    t_start = timeCount();
    warmup<<<grid, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int sum_gpu = 0;
    for(int i = 0; i < grid.x; i ++){
        sum_gpu += h_res_from_gpu[i];
        // printf("%d, %d, %d\n", i, h_res_from_gpu[0], h_res_from_gpu[i]);
    }
    
    printf("G=>warmup : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f\n", grid.x, block.x, sum_gpu, t_stop);
    
    InitialData(h_a, nElem);
    // memset(h_res_from_gpu, 0, grid.x * sizeof(int));
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    // CHECK(cudaMemset(d_res, 0 ,grid.x * sizeof(int)));
    t_start = timeCount();
    reduce_interger_unroll4<<<grid.x/4, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x / 4; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    float ibd = get_ibd(nElem, nElem, t_stop);
    printf("G=>reduce_interger_unroll4 : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x/4, block.x, sum_gpu, t_stop, ibd);

    InitialData(h_a, nElem);
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    t_start = timeCount();
    reduce_interger_complete_unroll4<<<grid.x/4, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x / 4; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    ibd = get_ibd(nElem, nElem, t_stop);
    printf("G=>reduce_interger_complete_unroll4 : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x/4, block.x, sum_gpu, t_stop, ibd);

    InitialData(h_a, nElem);
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    t_start = timeCount();
    reduce_interger_share<<<grid.x, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    ibd = get_ibd(nElem, nElem, t_stop);
    printf("S=>reduce_interger_share : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x, block.x, sum_gpu, t_stop, ibd);

    InitialData(h_a, nElem);
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    t_start = timeCount();
    reduce_interger_unroll4_share<<<grid.x/4, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x/4; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    ibd = get_ibd(nElem, nElem, t_stop);
    printf("S=>reduce_interger_unroll4_share : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x/4, block.x, sum_gpu, t_stop, ibd);

    InitialData(h_a, nElem);
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    t_start = timeCount();
    reduce_interger_complete_unroll4_share<<<grid.x/4, block>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x/4; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    ibd = get_ibd(nElem, nElem, t_stop);
    printf("S=>reduce_interger_complete_unroll4_share : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x/4, block.x, sum_gpu, t_stop, ibd);

    InitialData(h_a, nElem);
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    t_start = timeCount();
    reduce_interger_complete_unroll4_dynshare<<<grid.x/4, block, DIM*sizeof(int)>>>(d_a, d_res, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    CHECK(cudaMemcpy(h_res_from_gpu, d_res, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    sum_gpu = 0;
    for(int i = 0; i < grid.x/4; i ++){
        sum_gpu += h_res_from_gpu[i];
    }
    ibd = get_ibd(nElem, nElem, t_stop);
    printf("S=>reduce_interger_complete_unroll4_dynshare : execution configuration <<< %d, %d>>>, result is %d, kernel runtime is %.6f, effective bandwidth is %.2f GB \n", grid.x/4, block.x, sum_gpu, t_stop, ibd);

    free(h_a);
    free(h_res_from_gpu);
    cudaFree(d_a);
    cudaFree(d_res);

    return 0;
}