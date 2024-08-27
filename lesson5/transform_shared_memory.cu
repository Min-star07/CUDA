#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "../freshman.h"

#define BDIMX 32
#define BDIMY 32

#define NPAD 2

void transpose_matrix_cpu(int *in, int *out, int nx, int ny){
    for(int i = 0; i < ny; i ++){
        for(int j = 0; j < nx; j ++){
            out[i * nx + j] = in[j * ny + i];
        }
    }
}
void print_matrix(int *matrix, int nx, int ny, int size){
    for( int i = 0; i < size; i ++){
        printf("%d ", matrix[i]) ;
        if((i + 1) % nx == 0)
            printf("\n");
    }
}

__global__ void warmup(int *in, int *out, int nx, int ny, int size){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;

    int ti = iy * nx + ix;
    // int to = ix * ny + iy;
    if(ix < nx && iy< ny){
        out[ti] = in[ti];
    }
}
__global__ void copy_row(int *in, int *out, int nx, int ny, int size){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;

    int ti = iy * nx + ix;
    // int to = ix * ny + iy;
    if(ix < nx && iy< ny){
        out[ti] = in[ti];
    }
}

__global__ void transform_naiverow(int *in, int *out, int nx, int ny, int size){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;

    int ti = iy * nx + ix;
    int to = ix * ny + iy;
    if(ix < nx && iy< ny){
        out[to] = in[ti];
    }
}
__global__ void transform_row_smem(int *in, int *out, int nx, int ny, int size){
    __shared__ int tile[BDIMY][BDIMX];

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;
    int ti = iy * nx + ix;

    //block idx
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    // 
    int to = iy * ny + ix;
    if(ix < nx && iy< ny){
        // load data from global to shared
        tile[threadIdx.y][threadIdx.x] = in[ti];

        __syncthreads();

        // store data from shared memory to global memory
        out[to] = tile[icol][irow];
    }
}

__global__ void transform_row_smem_npad(int *in, int *out, int nx, int ny, int size){
    __shared__ int tile[BDIMY][BDIMX + NPAD];

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;
    int ti = iy * nx + ix;

    //block idx
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    // 
    int to = iy * ny + ix;
    if(ix < nx && iy< ny){
        // load data from global to shared
        tile[threadIdx.y][threadIdx.x] = in[ti];

        __syncthreads();

        // store data from shared memory to global memory
        out[to] = tile[icol][irow];
    }
}

__global__ void transform_row_smem_npad_unroll(int *in, int *out, int nx, int ny, int size){
    __shared__ int tile[BDIMY*(BDIMX*2) + NPAD];

    int ix = threadIdx.x + blockIdx.x * blockDim.x * 2;
    int iy = threadIdx.y + blockIdx.y * blockDim.x;
    int ti = iy * nx + ix;

    //block idx
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x * 2  + irow;
    
    if(ix < nx && iy< ny){
        // load 2 rows  from global to shared
        unsigned int row_idx = threadIdx.y *(blockDim.x * 2 + NPAD) + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in [ti + BDIMX];
        __syncthreads();

        // store data from shared memory to global memory
        unsigned int col_idx = icol * (blockDim.x * 2 +NPAD) + irow;
        out[col_idx] = tile[col_idx];
        out[col_idx + BDIMX] = tile[col_idx + BDIMX];
    }
}

int main(int argc, char** argv){
    int nx = 1 << 12;
    int ny = 1 << 12;
    int dimx = BDIMX;
    int dimy = BDIMY;
    // int transform_kernel = 0;
    if(argc > 1){
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }
    int nElem = nx * ny;
    printf("Size : %d\n", nElem);
    int nByte = nElem * sizeof(int);

    int *h_a = (int*)malloc(nByte);
    int *h_b = (int*)malloc(nByte);
    int *h_b_gpu = (int*)malloc(nByte);
    InitialData(h_a, nElem);
    // print_matrix(h_a, nx, ny, nElem);
    double t_start, t_stop;
    // result from cpu
    t_start = timeCount();
    transpose_matrix_cpu(h_a, h_b, nx, ny);
    // print_matrix(h_b, nx, ny, nElem);
    t_stop = timeCount() - t_start;
    printf("result in cpu run time is %.6f s\n" , t_stop);

    int *d_a, *d_b;
    CHECK(cudaMalloc((void**)&d_a, nByte));
    CHECK(cudaMalloc((void**)&d_b, nByte));

    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b_gpu, nByte, cudaMemcpyHostToDevice));

    dim3 block(dimx, dimy);
    dim3 grid((nElem + block.x + 1)/block.x, (nElem + block.y + 1)/block.y);
    t_start = timeCount();
    warmup<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("G=>warmup execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);

    t_start = timeCount();
    copy_row<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("G=>copy_row execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);
    t_start = timeCount();
    transform_naiverow<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("G=>transform_naiverow execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);

    // share memory
    t_start = timeCount();
    transform_row_smem<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("S=>transform_row_smem execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);
    t_start = timeCount();
    transform_row_smem_npad<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("S=>transform_row_smem_npad execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);
    t_start = timeCount();
    transform_row_smem_npad_unroll<<<grid, block>>>(d_a, d_b, nx, ny, nElem);
    cudaDeviceSynchronize();
    t_stop = timeCount() - t_start;
    printf("S=>transform_row_smem_npad_unroll execution configuration <<<(%d, %d), (%d, %d)>>> , run time is %.6f s\n" , grid.x, grid.y, block.x, block.y, t_stop);
    CHECK(cudaMemcpy(h_b_gpu, d_b, nByte, cudaMemcpyDeviceToHost));
    // print_matrix(h_b_gpu, nx, ny, nElem);
   return 0;
}