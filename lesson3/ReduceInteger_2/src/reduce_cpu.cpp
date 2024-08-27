#include "reduce.h"

int reduceOnCPU(int *data,  int nElem)
{
    if(nElem == 1)
        return data[0];
    int stride = nElem / 2;
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }
   
    return reduceOnCPU(data, stride);
}

