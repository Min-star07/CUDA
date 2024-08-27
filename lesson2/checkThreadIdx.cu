#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(call){\
const cudaError_t error = call;\
if (error!= cudaSuccess){\
printf("Error : %s : %d\n",  __FILE__, __LINE__);\
printf("code : %d, reason : %s\n", error, cudaGetErrorString(error));\
exit(-10*error);\
}\
}
void initialInt(int *ip, int size){
    for (int i = 0; i < size; i++){
        ip[i] = i;
    }
}

// void printMatrix(int *C, const int nx, const int ny){
//     int *ic = C;
//     printf("Matrix, nx: %d, ny: %d\n", nx, ny);
//     for(int i = 0; i < ny; i++){
//         for(int j = 0; j < nx; j++){
//             printf("%d ", *ic);
//             ic++;
//         }
//         printf("\n");
//     }
// }


void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("Matrix, nx: %d, ny: %d\n", nx, ny);
    for(int i = 0; i < ny; i++){
        for(int j = 0; j < nx; j++){
            printf("%3d ", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = col * nx + row;

    if(row < ny && col < nx){
        printf("Thread : (%d, %d)  block_id (%d, %d), coordinate (%d, %d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,  row, col, idx,  A[idx]);
    }
}
int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);
    //get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Dvice %d : %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set matrix dimensionj
    const int nx = 8;
    const int ny = 6;
    int size = nx * ny;
    int nBytes = size * sizeof(float);

    // allocate memory on device
    // allocate memory on host
    int *h_A;
    h_A = (int *) (malloc(nBytes));

    // initialize matrix with data
    initialInt(h_A, size);
    // print original matrix
    printf("Original Matrix:\n");
    printMatrix(h_A, nx, ny);

    // // allocate memory on device
    int *d_A;
    CHECK(cudaMalloc((void **) &d_A, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // // set up configuration
    dim3 block(4,2);
    dim3 grid((nx + block.x-1)/ block.x, (ny + block.y-1)/ block.y);


    // launch kernel
    printThreadIndex<<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();
    // free memory on device and host
    CHECK(cudaFree(d_A));
    free(h_A);

    printf("%s Done...\n", argv[0]);

    // reset device
    CHECK(cudaDeviceReset());
    return 0;
}

