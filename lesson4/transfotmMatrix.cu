#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "../freshman.h"
// const int seed =1;

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

void initMatrix(int *data, int size){
     time_t t;
	srand((unsigned) time(&t));
	for (int i = 0; i<size; i++)
	{
		// data[i] = int(rand()&0xff);
		data[i] = i;
	}
}

void transformMatrix(int *MatA, int *MatB, int nx, int ny){
    for(int j = 0; j < nx; j ++){
        for(int i = 0 ;i < ny; i ++){
            MatB[i * nx + j] = MatA[j * ny + i];
        }
    }
}

void printMatrix(int *Max, int nx, int ny){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){
            printf("%d\t", Max[i * ny + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void copyRow(int * MatA,int * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    // printf("%d %d %d %d %d\n", idx, ix, iy, MatA[idx], MatB[idx]);
    // printf("===================================\n");
    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y,   blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, idx, MatA[idx]);

    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
    // printf("============\n");
    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y,   blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, idx, MatA[idx]);
    // printf("%d, %d\n", idx, MatA[idx]);
}

__global__ void copyCol(int *MatA, int *MatB, int nx, int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = ix * ny + iy;

    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y,   blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, idx, MatA[idx]);

    if(ix < nx && iy < ny){
        MatB[idx] = MatA[idx];
    }
    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y,   blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, idx, MatA[idx]);
    // printf("%d, %d\n", idx, MatA[idx]);
}

__global__ void transposeNaiveRow(int * MatA,int * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if (ix<nx && iy<ny)
    {
      MatB[ix * ny + iy]=MatA[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(int * MatA,int * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if (ix<nx && iy<ny)
    {
      MatB[iy * nx + ix]=MatA[ix * ny + iy];
    }
}

__global__ void transposeUnroll4NaiveRow(int * MatA,int * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x * 4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    
    int idx_row = iy * nx + ix;
    int idx_col = ix * ny + iy;
    if (ix + 1 * blockDim.x < nx && iy<ny)
    {
    MatB[idx_col] = MatA[idx_row];  
    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y, idx_row, idx_col,  blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, MatA[idx_row], MatA[idx_col]);
    // printf("=======================\n");
    MatB[idx_col + ny * 1 * blockDim.x] = MatA[idx_row + 1 * blockDim.x];
    // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d), (%d, %d) %d, %d\n", ix, iy,   threadIdx.x, threadIdx.y, idx_row, idx_col,  blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, MatA[idx_row], MatA[idx_col]);

      MatB[idx_col + ny * 2 * blockDim.x] = MatA[idx_row + 2 * blockDim.x];
      MatB[idx_col + ny * 3 * blockDim.x] = MatA[idx_row + 3 * blockDim.x];
    }
}

__global__ void transposeUnroll4NaiveCol(int * MatA,int * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x * 4 ;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;

    int idx_row = ix * ny + iy;
    int idx_col = iy * nx + ix;
    if (ix<nx && iy + blockIdx.y < ny)
    {
      MatB[idx_row]=MatA[idx_col];
      MatB[idx_row + blockDim.x]=MatA[idx_col + ny * 1 * blockDim.x];
      MatB[idx_row + 2 * blockDim.x] = MatA[idx_col + ny * 2 * blockDim.x];
      MatB[idx_row + 3 * blockDim.x] = MatA[idx_col + ny * 3 * blockDim.x];
    }
}

__global__ void transposeDiagonalRow(int *MatA, int *MatB, int nx, int ny){
    int blk_y = blockIdx.x;
    int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int ix = blockDim.x * blk_x + threadIdx.x;
    int iy = blockIdx.y * blk_y + threadIdx.y;

    int idx_row = iy * nx + ix;
    int idx_col = ix * ny + iy;
    if(ix < nx && iy < ny){
        MatA[idx_col] = MatB[idx_row];
    }
}

__global__ void transposeDiagonalCol(int *MatA, int *MatB, int nx, int ny){
    int blk_y = blockIdx.x;
    int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int ix = blockDim.x * blk_x + threadIdx.x;
    int iy = blockIdx.y * blk_y + threadIdx.y;

    int idx_row = iy * nx + ix;
    int idx_col = ix * ny + iy;
    if(ix < nx && iy < ny){
        MatA[idx_row] = MatB[idx_col];
    }
}
int main(int argc, char** argv){
    setGPU();

    // Set up matrix size
    int nx = 1 << 12; 
    int ny = 1 << 12;
    int dimx = 32;
    int dimy = 32;
    int transform_kernel=0;
    
    // Setup kernel and block size
    if(argc == 2){
        transform_kernel = atoi(argv[1]);
    }
    if(argc == 4){
        transform_kernel=atoi(argv[1]);
        dimx=atoi(argv[2]);
        dimy=atoi(argv[3]);
    }
    if(argc > 4){
        transform_kernel=atoi(argv[1]);
        dimx=atoi(argv[2]);
        dimy=atoi(argv[3]);
        nx = atoi(argv[4]);
        ny = atoi(argv[5]);
    }
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);
    
    int *h_MaxA = (int*)malloc(nBytes);
    int *h_MaxB = (int*)malloc(nBytes);
    int *h_MaxC_from_GPU = (int*)malloc(nBytes);
    initMatrix(h_MaxA, nxy);
    transformMatrix(h_MaxA, h_MaxB, nx, ny);

    // printMatrix(h_MaxA, nx, ny);
    // printMatrix(h_MaxB, nx, ny);

    int *d_MaxA=NULL, *d_MaxB=NULL;
    CHECK(cudaMalloc((void**)&d_MaxA, nBytes));
    CHECK(cudaMalloc((void**)&d_MaxB, nBytes));

    CHECK(cudaMemcpy(d_MaxA, h_MaxA, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_MaxB, 0, nBytes));  // Iniidx_rowalize device output memory

    // 2D block and grid configuraidx_rowon
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/ block.x, (ny + block.y - 1) / block.y);

    dim3 gridunroll((nx + block.x * 4 - 1)/ block.x*4, (ny + block.y - 1) / block.y);
    
    double iStart, iStop;
    iStart = timeCount();
    const char *kernel_name;
    switch (transform_kernel){
        case 0:
            copyRow<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "copyRow";
            break;

        case 1:
            copyCol<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "copyCol";
            break;
        case 2:
            transposeNaiveRow<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeNaiveRow";
            break;
        case 3:
            transposeNaiveCol<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeNaiveCol";
            break;
        case 4:
            transposeUnroll4NaiveRow<<<gridunroll, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeUnroll4NaiveRow";
            break;
        case 5:
            transposeUnroll4NaiveRow<<<gridunroll, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeUnroll4NaiveCol";
            break;
        case 6:
            transposeUnroll4NaiveRow<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeDiagonalRow";
            break;
        case 7:
            transposeUnroll4NaiveRow<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
            kernel_name = "transposeDiagonalCol";
            break;
        default:
            break;
    }
    // copyRow<<<grid, block>>>(d_MaxA, d_MaxB, nx, ny);
    cudaDeviceSynchronize();
    iStop = timeCount() - iStart;
    float ibnd = 2 * nx * ny * sizeof(int) / 1e9 / iStop;
    printf("%s : executioon configuration <<<(%d, %d), (%d, %d)>>> ; consuming time is  %.6f s ; effective bandwith : %f GB\n", kernel_name, grid.x, grid.y, block.x, block.y, iStop, ibnd);
    // calculate effecidx_rowve bandwidth
    
    CHECK(cudaMemcpy(h_MaxC_from_GPU, d_MaxB, nBytes, cudaMemcpyDeviceToHost));
    // printMatrix(h_MaxC_from_GPU, nx, ny);
    
    free(h_MaxA);
    free(h_MaxB);
    free(h_MaxC_from_GPU);
    cudaFree(d_MaxA);
    cudaFree(d_MaxB);

    return 0;
}
