#include "utils.h"
#include <math.h>
#include <random>

void initMatrix(int *data, int size, int seed){
    srand(seed);
    for (int i = 0; i < size; i++) {
        // data[i] = int(rand()/RAND_MAX);
        data[i] = i;
        // mask off high 2 bytes to force max number 255
        // data[i] = (int)(rand() & 0xFF);
    }
}

void printMat(int *data, int size){
    for (int i = 0; i < size; i++) {
        printf("%d", data[i]);
        if(i != size-1)
            printf(",");
        else
            printf("\n");      
    }
}

void compareMat(int* h_data, int* d_data, int size) 
{
    double precision = 1.0E-4;
    for (int i = 0; i < size; i ++) {
        if (abs(h_data[i] - d_data[i]) > precision) {
            int y = i / size;
            int x = i % size;
            printf("Matmul result is different\n");
            printf("cpu: %d, gpu: %d, cord:[%d, %d]\n", h_data[i], d_data[i], x, y);
            break;
        }
    }
}

int sumArraysOnGPU(int *h_odata, int size){
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += h_odata[i];
        // printf("%d, %d\n", i, h_odata[i]);
    }
    return sum;
}