#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <algorithm>

#define u32 unsigned int
#define u64 unsigned long
#define uchar unsigned char
#define BLOCK_SIZE 64
#define FULL_MASK 0xffffffff

/** !ISSUE
 * sometimes sorted array is partially change, why?
 * e.g.
 * CPU SORT: 1, 2, 3, 3, 4, 5
 * GPU SORT: 1, 2, 2, 3, 4, 5
 */

#define CREATE_RAND_ARR(arr, size, min, max) \
do {                                         \
    time_t t;                                \
    srand((unsigned)time(&t));               \
    for (u32 i = 0; i < size; i++)           \
        arr[i] = rand() % max + min;         \
} while (0)                                  \

#define COPY_ARR(a, b, size)       \
do {                               \
    for (u32 i = 0; i < size; i++) \
        a[i] = b[i];               \
} while (0)                        \

#define PRINT_ARR(arr, size)       \
do {                               \
    for (u32 i = 0; i < size; i++) \
        printf("%u, ", arr[i]);    \
    printf("\n");                  \
} while (0)                        \

#define ARR_EQ(a, b, size)           \
do {                                 \
    bool eq = true;                  \
    for (u32 i = 0; i < size; i++) { \
        if (a[i] != b[i]) {          \
            eq = false;              \
            break;                   \
        }                            \
    }                                \
    if (eq)                          \
        printf("EQUAL\n");           \
    else                             \
        printf("NOT EQAUL\n");       \
} while (0)                          \

void check(const u32* a, const u32 size)
{
    bool correct = true;
    for (u32 i = 0; i < size - 1; i++) {
        if (a[i] > a[i + 1]) {
            printf("%u, %u\n", a[i] , a[i + 1]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("CORRECT\n");
    }
    else {
        printf("NOT CORRECT\n");
    }
}

int compare(const void* a, const void* b)
{
    int c = *(int*)a;
    int d = *(int*)b;
    if(c < d) {return -1;}
    else if (c == d) {return 0;}
    else return 1;
}

__global__
void global_bubble_sort_kernel(const u32* __restrict__ arr, const u32 size, u32* sorted_arr)
{
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x >= size) {
        return;
    }

    sorted_arr[id_x] = arr[id_x];
    __syncthreads();
    int key = 0;
    for (u32 n = size * 2; n > 0; n--) {
        if ((id_x & 1) == key && (id_x < size - 1)) {
            if (sorted_arr[id_x] > sorted_arr[id_x + 1]) {
                // sorted_arr[id_x] = sorted_arr[id_x + 1];
                // memcpy(sorted_arr + id_x + 1, &temp, sizeof(u32));
                // atomicExch(&sorted_arr[id_x], sorted_arr[id_x + 1]);
                u32 temp1 = sorted_arr[id_x];
                u32 temp2 = sorted_arr[id_x + 1];
                atomicCAS(&sorted_arr[id_x], temp1, sorted_arr[id_x + 1]);
                atomicCAS(&sorted_arr[id_x + 1], temp2, temp1);
            }
        }
        key ^= 1;
        __syncthreads();
    }
}

void gpuGlobalBubbleSort(const u32* h_arr, const u32 size, u32* h_sorted_arr)
{
    assert(size > 0);

    u32* d_arr;
    cudaMalloc((void**)&d_arr, size * sizeof(u32));
    cudaMemcpy(d_arr, h_arr, size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_sorted_arr;
    cudaMalloc((void**)&d_sorted_arr, size * sizeof(u32));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    global_bubble_sort_kernel <<< grids, blocks >>> (d_arr, size, d_sorted_arr);
    if (cudaSuccess != cudaGetLastError()) {
        printf("global_bubble_sort_kernel fault!\n");
    }

    cudaMemcpy(h_sorted_arr, d_sorted_arr, size * sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_sorted_arr);
}

/**
 * @ref https://github.com/Kshitij421/Odd-Even-Sort-using-Cuda-/blob/master/oddeven.cu
 * @ref https://github.com/znahman/parallel-comparison-bubblesort/blob/master/parallel-bubsort.cu
 */
int main()
{
    const u32 size = 10000;
    u32* arr = (u32*)malloc(size * sizeof(u32));
    CREATE_RAND_ARR(arr, size, 0, 10000);

    // PRINT_ARR(arr, size);
    
    u32* cpuSortArr = (u32*)malloc(size * sizeof(u32));
    COPY_ARR(cpuSortArr, arr, size);
    qsort((void*)cpuSortArr, size, sizeof(u32), compare);
    // PRINT_ARR(cpuSortArr, size);

    u32* gpuGlobalBubbleSortArr = (u32*)malloc(size * sizeof(u32));
    gpuGlobalBubbleSort(arr, size, gpuGlobalBubbleSortArr);
    // PRINT_ARR(gpuGlobalBubbleSortArr, size);
    check(gpuGlobalBubbleSortArr, size);

    // ARR_EQ(gpuGlobalBubbleSortArr, cpuSortArr, size);

    free(arr);
    free(cpuSortArr);
    free(gpuGlobalBubbleSortArr);
    
    return 0;
}