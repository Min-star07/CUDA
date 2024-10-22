#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;
// using cooperative_groups::thread_group; // etc.

__global__ void cooperativeKernel(){
    auto block = this_thread_block();
    // Get the size of the thread group
    unsigned group_size = block.size();

    // Get the thread rank in the group
    unsigned rank = block.thread_rank();

    // Check if group is valid
    // if(block.is_valid()){
    //     printf("Thread %u of %u in block %u\n", rank, group_size, blockIdx.x);
    // }
    printf("Thread %u of %u in block %u\n", rank, group_size, blockIdx.x);
    // Synchronize threads within the group
    block.sync();


}

int main() {
    // Launch a kernel with 2 blocks of 8 threads each
    cooperativeKernel<<<2, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}