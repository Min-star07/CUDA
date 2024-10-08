#include<cuda_runtime.h>
#include<stdio.h>

#define DSIZE 1024
#define BLOCK_SIZE 32

void sum_cpu_res(int *A, int *sums, int ds){
    sums[0] = 0;
    for(int i =0; i < ds; i++){
        sums[0] +=A[i];
    }
    printf("%d\n", sums[0]);
}

__global__ void sum_rows(int *A, int *sums, int ds){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int sum = 0;
    if (idx < ds && idy < ds){
        // atomicAdd(sums, A[idy * ds + idx]);
        for(int i =0; i < ds; i++){
            sum += A[idy * ds + i];
        }
        sums[idx] = sum;
    }
    __syncthreads();
}

__global__ void sum_columns(int *A, int *sums, int ds){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < ds && idy < ds){
       atomicAdd(sums, A[idx * ds + idy]);
    }
    __syncthreads();
}

void setInit(int *mat, int size){
    for(int i =0; i < size; i++){
        mat[i] = 1;
    }
}

void diff_compare(int *A, int *B){
    if(A[0]-B[0] > 1e-6){
        printf("no Match : %d, %d\n", A[0], B[0]);
    }
    else{
        printf("Match : %d, %d\n", A[0], B[0]);
    }
}
int main(int argc, char** argv){
    int *A_h,  *A_d ;
    int *sum_cpu, *sum_row_gpu, *sum_column_gpu, *sum_row, *sum_column;

    int nBytes = DSIZE * DSIZE* sizeof(int);
  
    // void *malloc(size_t size);
    A_h = (int*)malloc(nBytes);
    sum_cpu = (int*)malloc(sizeof(int));
    sum_column_gpu = (int*)malloc(sizeof(int));
    sum_row_gpu = (int*)malloc(sizeof(int));

    setInit(A_h, DSIZE*DSIZE);

    sum_cpu_res(A_h, sum_cpu, DSIZE*DSIZE);

    cudaMalloc((void**)&A_d, nBytes);
    cudaMalloc((void**)&sum_column, sizeof(int));
    cudaMalloc((void**)&sum_row, sizeof(int));

    cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((BLOCK_SIZE + DSIZE -1)/block.x, (BLOCK_SIZE + DSIZE -1)/block.y);

    sum_rows<<<grid, block>>>(A_d, sum_row, DSIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(sum_row_gpu, sum_row, sizeof(int), cudaMemcpyDeviceToHost);
    sum_columns<<<grid, block>>>(A_d, sum_column, DSIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(sum_column_gpu, sum_column, sizeof(int), cudaMemcpyDeviceToHost);

    diff_compare(sum_row_gpu, sum_column_gpu);

    free(A_h);
    free(sum_cpu);
    free(sum_row_gpu);
    free(sum_column_gpu);
    cudaFree(sum_row);
    cudaFree(sum_column);
    cudaFree(A_d);

    return 0;
}