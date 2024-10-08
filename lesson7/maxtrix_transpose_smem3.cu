#include <cuda_runtime.h>
#include <stdio.h>

#define M 3000
#define N 1000
#define BLOCK_SIZE 32

__managed__ int A[M][N];
__managed__ int B_cpu[N][M];
__managed__ int B_gpu[N][M];

void setInit(int A[M][N]){
    for(int y = 0; y < M; y ++){
        for(int x = 0; x < N; x++)
        A[y][x] =rand()%1024;
    }

}

void maxtrix_transpose_cpu(int A[M][N], int B[N][M]){
    for(int y =0; y < M; y++){
        for(int x =0; x < N; x++){
            B[x][y] = A[y][x];
        }
    }

}

__global__ void matrix_trans_smem_gpu(int A[M][N], int B[N][M]){

    __shared__ int As[BLOCK_SIZE+1][BLOCK_SIZE+1];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < N && idy < M){
        As[threadIdx.y][threadIdx.x] = A[idy][idx];
    }
    __syncthreads();
    int trans_idx = threadIdx.x + blockIdx.y * blockDim.y;
    int trans_idy = threadIdx.y + blockIdx.x * blockDim.x;

    if(trans_idx < M && trans_idy < N){
        B[trans_idy][trans_idx] = As[threadIdx.x][threadIdx.y];
    }


}

void diff_compare(int res_cpu[N][M], int res_gpu[N][M]){
    for(int y = 0; y < N; y ++){
        for(int x =0; x < M; x++){
            if(res_cpu[y][x] - res_gpu[y][x] > 1e-6)
            {printf("Mismatch %d, %d, CPU %d, GPU %d\n", y, x, res_cpu[y][x], res_gpu[y][x]);}
            // {printf("Mismatch %d, %d, CPU %d, GPU %d\n", y, x, res_cpu[y][x], res_gpu[y][x]);}
        }
        
         
    }
    printf("CPU and GPU results match.\n");
}

int main(int argc, char** argv){

    setInit(A);
    maxtrix_transpose_cpu(A, B_cpu);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x -1)/block.x, (M+block.y -1)/block.y);

    matrix_trans_smem_gpu<<<grid, block>>>(A,B_gpu);

    cudaDeviceSynchronize();

    diff_compare(B_cpu,B_gpu);

    return 0;
}