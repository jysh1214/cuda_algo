#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define CREATE_RAND_ARR(arr, size, min, max) \
do {                                         \
    time_t t;                                \
    srand((unsigned)time(&t));               \
    for (int i = 0; i < size; i++)           \
        arr[i] = rand() % max + min;         \
} while (0)                                  \

#define PRINT_ARR(arr, size)       \
do {                               \
    for (int i = 0; i < size; i++) \
        printf("%u, ", arr[i]);    \
    printf("\n");                  \
} while (0)                        \

__global__ void ballot_kernel(int* before, int* after, const int size)
{
   int threadsmask = 0xffffffff << threadIdx.x;
   if (threadIdx.x < size) {
        int e = threadIdx.x & 1;
        // before[threadIdx.x] = e; 
        int ones = __ballot_sync(0xffffffff, e);
        after[threadIdx.x] = ones;
        before[threadIdx.x] = __popc(ones & threadsmask);
   }
}

int main()
{
    const int size = 8;

    int* d_after;
    int* d_before;
    cudaMalloc((void**)&d_after, size * sizeof(int));
    cudaMalloc((void**)&d_before, size * sizeof(int));

    ballot_kernel <<< 1, 32 >>> (d_before, d_after, size);
    
    int* h_before = (int*)malloc(size * sizeof(int));
    int* h_after = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_before, d_before, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_after, d_after, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("BEFORE:\n");
    PRINT_ARR(h_before, size);
    printf("AFTER:\n");
    PRINT_ARR(h_after, size);

    cudaFree(d_after);
    cudaFree(d_before);
    free(h_before);
    free(h_after);
    
    return 0;
}