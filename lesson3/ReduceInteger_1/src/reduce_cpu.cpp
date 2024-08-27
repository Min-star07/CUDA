#include "reduce.h"

int ReduceOnCPU(int *data, int const size)
{
    int sum = 0;
    for (int i = 0; i < size; ++i){
        sum += data[i];
    }
    return sum;
}