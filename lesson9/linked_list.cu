// This code creates a simple linked list of elements in managed memory 
// that can be accessed by both the CPU (host) and the GPU (device), 
// and it includes error-checking mechanisms for CUDA operations. 
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>  // Include for CUDA functions

// Error-checking macro for CUDA calls
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

struct list_elem {
    int key;
    list_elem *next;
};

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes) {
    cudaError_t err = cudaMallocManaged(&ptr, num_bytes);  // Use cudaMallocManaged for unified memory
    cudaCheckErrors("CUDA memory allocation failed!");     // Check for allocation errors
}

__host__ __device__
void print_element(list_elem *list, int ele_num) {
    list_elem *elem = list;
    for (int i = 0; i < ele_num; i++) {
        elem = elem->next;  // Traverse to the next element
        if (elem == nullptr) {
            printf("Reached the end of the list. No more elements.\n");
            return;
        }
        printf("key = %d\n", elem->key);  // Print the key of the desired element
    }
    
}

__global__ void gpu_print_element(list_elem *list, int ele_num) {
    print_element(list, ele_num);  // Call print function on GPU
}

const int num_elem = 5;  // Number of elements in the list
const int ele = 3;       // Element index to print

int main() {
    list_elem *list_base, *list;

    // Allocate memory for the base of the list
    alloc_bytes(list_base, sizeof(list_elem));
    printf("Size of list_elem: %zu bytes\n", sizeof(list_elem));

    list = list_base;

    // Initialize the linked list with num_elem elements
    for (int i = 0; i < num_elem; i++) {
        list->key = i;  // Set key for the current element
        if (i == num_elem - 1) {
            list->next = nullptr;  // Set next to nullptr for the last element
        } else {
            alloc_bytes(list->next, sizeof(list_elem));  // Allocate memory for the next element
        }
        list = list->next;  // Move to the next element
    }

    // Print the element from the CPU
    // print_element(list_base, ele);

    // Launch the kernel to print the element from the GPU
    gpu_print_element<<<1, 1>>>(list_base, ele);
    cudaDeviceSynchronize();  // Wait for the GPU to finish
    cudaCheckErrors("CUDA error after kernel execution!");

    return 0;
}
