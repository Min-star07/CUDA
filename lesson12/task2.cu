#include <stdio.h>
#ifdef DEBUG
#define CUDA_CALL(F) if((F) != cudaSuccess)\
{\
    printf("Error : %s at %s : %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
    exit(-1);
}
#define CUDA_CHECK() if((cudaPeekAtLastError()) != cudaSuccess){\
    printf("Error : %s at %s : %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
    exit(-1)\
}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

// define threadblock size in x and Y
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

// define matrix linew dimension
#define SIZE 4096

/* macro to index a 1D memory array with 2D indices in column-major order */
#define INDX(row, col, ld)(((col) * (ld)) + (row))

__global__ void smem_cuda_transpose(const int m, const double *const a, double * const c){
    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    /*declare a share memory*/
    __shared__ double smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    /*define my row and column tile index*/
    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if(myRow < m && myCol < m){
        // c[INDX(myRow, myCol, m)] = a[INDX(myCol, myRow, m)];
        smemArray[threadIdx.x][threadIdx.y] = a[INDX(tileX +threadIdx.x, tileY + threadIdx.y, m)];
    }
    __syncthreads();
    if(myRow<m &&myCol<m){
      c[INDX(tileY + threadIdx.x, tileX + threadIdx.y, m)] = smemArray[threadIdx.y][threadIdx.x];
    }
    return ;
}

void host_transpose(const int m, const double *a, double *c){
    for(int j =0; j <m; j++){
        for(int i =0;i <m;i ++){
            c[INDX(i,j,m)] = a[INDX(j, i,m)];
        }
    }
}


int main(int argc, char** argv){
    const int size = SIZE;
    fprintf(stdout, "Matrix size is : %d\n", size);

    double *h_a, *h_c;
    double *d_a, *d_c;

    size_t numbytes = (size_t) size * (size_t) size * sizeof(double);

    h_a = (double*)malloc(numbytes);
    if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc h_a\n");
    return 911;
  }

  h_c = (double *) malloc( numbytes );
  if( h_c == NULL )
  {
    fprintf(stderr,"Error in host malloc h_c\n");
    return 911;
  }
  /* allocating device memory */
  CUDA_CALL(cudaMalloc((void**) &d_a, numbytes));
  CUDA_CALL(cudaMalloc((void**) &d_c, numbytes));

  /*ser result martrix is zeros*/
  memset(h_a, 0, numbytes);
  CUDA_CALL(cudaMemset(d_c, 0, numbytes));

  fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
     (double) numbytes / 1000000.0 );

  /* initialize input matrix with random value */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  }

  /*copy input matrix from host to device*/
  CUDA_CALL(cudaMemcpy(d_a, h_a, numbytes, cudaMemcpyHostToDevice));

  /*create and start timer*/
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));
  CUDA_CALL(cudaEventRecord(start, 0));

  /*call naive cpu transpose function*/
  host_transpose(size, h_a, h_c);
  /*stop cpu timer*/
  CUDA_CALL( cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));

  float elapsedTime;
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

  fprintf(stdout, "Total time cpu is %f sec\n", elapsedTime/1000.0f);
  fprintf(stdout, "Performance is %f GB/s \n", 8.0 *2.0 *(double) size/((double)elapsedTime/1000.0)*1e-9);
  /*
  Breakdown of the formula:
    8.0 * 2.0 * (double)size * (double)size:

    8.0: Each double-precision number is 8 bytes.
    2.0: Implies that two memory operations are involved (such as reading and writing or some dual-buffered operation).
    (double)size * (double)size: Represents the total amount of data being processed. If size refers to the dimensions of a matrix or array, this would be the total number of elements (i.e., size^2).
    / ( (double) elapsedTime / 1000.0 ):

    This term converts the elapsed time from milliseconds to seconds by dividing elapsedTime by 1000.0. The operation measures how long it took to transfer the data or complete the operation in seconds.
    * 1.e-9:

    The multiplication by 1.e-9 converts the result to gigabytes per second (GB/s), since the data is currently measured in bytes and the time in seconds.
    The division by 1.e9 accounts for the conversion from bytes to gigabytes (1 GB = 10^9 bytes).
    */
    /* setup threadblock size and grid sizes */
  dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_X,1);
  dim3 blocks((size /THREADS_PER_BLOCK_X+1), (size/THREADS_PER_BLOCK_Y+1),1);

  /*start timer*/
  CUDA_CALL(cudaEventRecord(start ,0));
  /*kernel launch*/
  smem_cuda_transpose<<<blocks, threads>>>(size, d_a, d_c);
  /*stop timer*/
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));

  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

  /* print GPU timing information */

  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

  /*copy data from device to host*/
  CUDA_CALL( cudaMemset( d_a, 0, numbytes ) );
  CUDA_CALL(cudaMemcpy(h_a, d_c, numbytes, cudaMemcpyDeviceToHost));
  /* compare GPU to CPU for correctness */

  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
      {
        printf("Error in element %d,%d\n", i,j );
        printf("Host %f, device %f\n",h_c[INDX(i,j,size)],
                                      h_a[INDX(i,j,size)]);
        printf("FAIL\n");
        goto end;
      } /* end fi */
    } /* end for i */
  } /* end for j */

/* free the memory */
  printf("PASS\n");

  end:
  free( h_a );
  free( h_c );
  CUDA_CALL( cudaFree( d_a ) );
  CUDA_CALL( cudaFree( d_c ) );
  CUDA_CALL( cudaDeviceReset() );

  return 0;

}