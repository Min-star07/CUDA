#include <cuda_runtime.h>
#include <stdio.h>
#include "../freshman.h"

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16

#define IPAD 1
#define NPAD 2

__global__ void warmup(int *out){
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_row_read_row(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_col_read_col(int *out){
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_col_read_row(int *out){
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_row_read_col(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_row_read_col_dyn(int *out){
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.x + blockIdx.y * blockDim.x;
    unsigned int col_idx = threadIdx.y + blockIdx.x * blockDim.y;

    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

__global__ void set_row_read_col_ipad(int *out){
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_row_read_col_dyn_ipad(int *out){
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.x + blockIdx.y * (blockDim.x + IPAD);
    unsigned int col_idx = threadIdx.y + blockIdx.x * (blockDim.x + IPAD);
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[row_idx] = g_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

__global__ void set_row_read_col_rect(int *out){
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row_idx = idx / blockDim.y;
    unsigned int col_idx = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[col_idx][row_idx];
}
__global__ void set_row_read_col_rect_dyn(int *out){
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;

    unsigned int col_idx = icol * blockDim.x + irow;

    tile[idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}
__global__ void set_row_read_col_rect_ipad(int *out){
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT + NPAD];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row_idx = idx / blockDim.y;
    unsigned int col_idx = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[col_idx][row_idx];
}

__global__ void set_row_read_col_rect_dyn_ipad(int *out){
    extern __shared__ int tile[];
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col);
    unsigned int icol = g_idx % blockDim.y;
    unsigned int irow = g_idx / blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + NPAD) + threadIdx.x;

    // comvert back to smem idx to access the transposed element;
    unsigned int col_idx = icol * (blockDim.x +NPAD) + irow;

    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}

int main(int argc,char** argv){
    setGPU();
    int kernel = 0;
    if(argc>=2){
        kernel = atoi(argv[1]);
    }

    int nElem = BDIMX * BDIMY;
    int nBytes = nElem * sizeof(int);
    printf("Vector size : %d\n", nElem);

    int *out;

    CHECK(cudaMalloc((void**)&out, nBytes));

    cudaSharedMemConfig MemConfig;
    CHECK((cudaDeviceGetSharedMemConfig(&MemConfig)));
    printf("----------------------------------------------------\n");
    switch(MemConfig){
        case cudaSharedMemBankSizeFourByte:
            printf("The device id cudaSharedMemBankSizeFourByte: 4-Byte\n");
            break;
        case cudaSharedMemBankSizeEightByte:
            printf("The device id cudaSharedMemBankSizeEightByte: 8-Byte\n");
            break;
    }
    printf("----------------------------------------------------\n");

    /*start read the data according to different configuration*/
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1,1);

    dim3 block_rect(BDIMX_RECT, BDIMY_RECT);
    dim3 grid_rect(1,1);

    warmup<<<block, grid>>>(out);

    printf("Only warmup!!!\n");

    /*write global index into 2d shared memory by set row major order,
     then according to row order, save the data to global memory from shared memory*/
    double t_Start, t_Stop;
    const char* kernel_name;
    t_Start = timeCount();
    switch(kernel){
        case 0:
            set_row_read_row<<<grid, block>>>(out);
            kernel_name = "set_row_read_row";
            break;
        case 1:
            set_col_read_col<<<grid, block>>>(out);
            kernel_name = "set_col_read_col";
            break;
        case 2:
            set_row_read_col<<<grid, block>>>(out);
            kernel_name = "set_row_read_col";
            break;
        case 3:
            set_col_read_row<<<grid, block>>>(out);
            kernel_name = "set_col_read_row";
            break;
        case 4:
            set_row_read_col_dyn<<<grid, block, BDIMX *BDIMY * sizeof(int)>>>(out);
            kernel_name = "set_row_read_col_dyn";
            break;
        case 5:
            set_row_read_col_ipad<<<grid, block>>>(out);
            kernel_name = "set_row_read_col_ipad";
            break;
        case 6:
            set_row_read_col_dyn_ipad<<<grid, block, (BDIMX + IPAD) * BDIMY + 1 * sizeof(int)>>>(out);
            kernel_name = "set_row_read_col_dyn_ipad";
            break;
        case 7:
            set_row_read_col_rect<<<grid_rect, block_rect>>>(out);
            kernel_name = "set_row_read_col_rect";
            break;
        case 8:
            set_row_read_col_rect_dyn<<<grid_rect, block_rect, BDIMX_RECT * BDIMY_RECT *sizeof(int)>>>(out);
            kernel_name = "set_row_read_col_rect_dyn";
            break;
        case 9:
            set_row_read_col_rect_ipad<<<grid_rect, block_rect>>>(out);
            kernel_name = "set_row_read_col_rect_ipad";
            break;
        case 10:
            set_row_read_col_rect_dyn<<<grid_rect, block_rect, (BDIMX_RECT+NPAD) * BDIMY_RECT *sizeof(int)>>>(out);
            kernel_name = "set_row_read_col_rect_dyn_npad";
            break;
    } 
    t_Stop = timeCount() - t_Start;
    printf("%s : execution configuration <<<(%d, %d), (%d, %d)>>>, time consumption : %.6f s\n", kernel_name, grid.x, grid.y, block.x, block.y, t_Stop);
    
    cudaFree(out);
    return 0;
}