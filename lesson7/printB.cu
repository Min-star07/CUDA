#include <stdio.h>

#define block_size 2  // Size of each block (16x16 threads)

__global__ void matrixMulWithPrint(const float *A, const float *B, float *C, int ds) {
    // Calculate global row and column for each thread
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // Column index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // Row index

    // Declare shared memory for the sub-matrices of B
    __shared__ float Bs[block_size][block_size];

    // Ensure we are within bounds of the matrices
    if (idx < ds && idy < ds) {
        // Initialize a variable to accumulate the partial result for C[idy][idx]
        float temp = 0.0f;

        // Loop over tiles of the input matrices
        for (int i = 0; i < ds / block_size; i++) {
            // Global row and column of matrix B to load
            int globalRow = i * block_size + threadIdx.y; // The row in B to load
            int globalCol = idx;                          // The column in B to load

            // Check bounds to avoid reading out of bounds
            if (globalRow < ds && globalCol < ds) {
                // Load the value of matrix B into shared memory and print it before loading
                float bValue = B[globalRow * ds + globalCol];
                printf("B[%d][%d] = %f\n", globalRow, globalCol, bValue);
                printf("=================");

                // Load into shared memory
                Bs[threadIdx.y][threadIdx.x] = bValue;
            } else {
                // If out of bounds, set the value to 0
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Synchronize to ensure all threads have loaded the tile into shared memory
            __syncthreads();

            // Print shared memory Bs after loading
            printf("Bs[%d][%d] = %f\n", threadIdx.y, threadIdx.x, Bs[threadIdx.y][threadIdx.x]);

            // Perform the multiplication for this tile
            for (int k = 0; k < block_size; k++) {
                temp += Bs[threadIdx.y][k] * Bs[k][threadIdx.x];
            }

            // Synchronize again before loading the next tile
            __syncthreads();
        }

        // Write the computed result to global memory
        C[idy * ds + idx] = temp;
    }
}

int main() {
    int ds = 4;  // Size of the matrices (16x16 for simplicity)
    
    // Host matrices
    float h_A[ds * ds], h_B[ds * ds], h_C[ds * ds];

    // Initialize matrices A and B on host (for simplicity, we fill them with sequential values)
    for (int i = 0; i < ds * ds; i++) {
        h_A[i] = i * 0.1f;  // Initialize A with values 0.0, 0.1, 0.2, ...
        h_B[i] = i * 0.2f;  // Initialize B with values 0.0, 0.2, 0.4, ...
    }

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, ds * ds * sizeof(float));
    cudaMalloc((void**)&d_B, ds * ds * sizeof(float));
    cudaMalloc((void**)&d_C, ds * ds * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, ds * ds * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ds * ds * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threads(block_size, block_size); // block_size x block_size threads per block
    dim3 grid((ds + block_size - 1) / block_size, (ds + block_size - 1) / block_size); // grid to cover entire matrix

    // Launch the kernel
    matrixMulWithPrint<<<grid, threads>>>(d_A, d_B, d_C, ds);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, ds * ds * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting matrix C
    printf("Matrix C (Result of A x B):\n");
    for (int i = 0; i < ds; i++) {
        for (int j = 0; j < ds; j++) {
            printf("%f ", h_C[i * ds + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
