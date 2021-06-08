#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>

#define u32 unsigned int
#define u64 unsigned long
#define uchar unsigned char
#define BLOCK_SIZE 64
#define FULL_MASK 0xffffffff

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

#define IS_POWER_OF_2(n)       \
(n > 0 && (n && (n - 1) == 0)) \

#define ZERO_PADDING(arr, a, b) \
do {                            \
    for (u32 i = a; i < b; i++) \
        arr[i] = 0;             \
} while (0)                     \

static inline
u32 getNextPowOf2(const u32 num)
{
    u32 new_num = 1;
    if (!IS_POWER_OF_2(num)) {
        while (new_num < num) {
            new_num <<= 1;
        }
    }
    else {
        new_num = num;
    }

    return new_num;
}

__global__ 
void shared_reduce_sum_kernel(const u32* arr, const u32 size, u32* goal)
{
    extern __shared__ u32 shared_arr[];
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < size) {
        shared_arr[threadIdx.x] = arr[id_x];
        __syncthreads();
        for (u32 offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                shared_arr[threadIdx.x] += shared_arr[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            goal[blockIdx.x] = shared_arr[threadIdx.x];
        }
    }
}

__global__ 
void large_shared_reduce_sum_kernel(const u32* arr, const u32 size, u32* sum)
{
    extern __shared__ u32 shared_arr[];
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (id_x < size) {
        if (id_x == 0) {
            *sum = 0;
        }
        shared_arr[id_x] = arr[id_x];
        __syncthreads();
        for (u32 offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                shared_arr[id_x] += shared_arr[id_x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum, shared_arr[id_x]);
        }
    }
}

__global__ 
void global_reduce_sum_kernel(u32* arr, const u32 size, u32* sum)
{
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x >= size) {
        return;
    }
    for (u32 offset = size >> 1; offset > 0; offset >>= 1) {
        if (id_x < offset) {
            arr[id_x] += arr[id_x + offset]; 
        }
        __syncthreads();
    }

    *sum = arr[0];
}

__global__
void warp_reduce_sum_kernel(u32* arr, u32* sum)
{
    u32 val = arr[threadIdx.x];
    for (u32 offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
        arr[threadIdx.x] = val;
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        *sum = arr[threadIdx.x];
    }
}

__host__
u32 gpuWarpLevelReduce(const u32* const h_arr, const u32 size)
{
    assert(size == 32 /*warp size*/);

    u32* d_arr;
    cudaMalloc((void**)&d_arr, size * sizeof(u32));
    cudaMemcpy(d_arr, h_arr, size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_sum;
    cudaMalloc((void**)&d_sum, 1 * sizeof(u32));

    warp_reduce_sum_kernel <<< 1, 32 >>> (d_arr, d_sum);
    if (cudaSuccess != cudaGetLastError()) {
        printf("shared_reduce_sum_kernel fault!\n");
    }
    
    u32* h_sum = (u32*)malloc(1 * sizeof(u32));
    cudaMemcpy(h_sum, d_sum, 1 * sizeof(u32), cudaMemcpyDeviceToHost);
    u32 sum = *h_sum;

    cudaFree(d_arr);
    cudaFree(d_sum);
    free(h_sum);

    return sum;
}

/**
 * use shared memory the size as block size
 * merge twice
 */
__host__
u32 gpuSharedReduceSum(const u32* const h_arr, const u32 size)
{
    assert(size > 0);

    u32 impl_size = getNextPowOf2(size);
    u32* temp = (u32*)malloc(impl_size * sizeof(u32));
    COPY_ARR(temp, h_arr, size);
    ZERO_PADDING(temp, size, impl_size);
    
    u32* d_arr;
    cudaMalloc((void**)&d_arr, impl_size * sizeof(u32));
    cudaMemcpy(d_arr, temp, impl_size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_sum;
    cudaMalloc((void**)&d_sum, 1 * sizeof(u32));

    u32* h_sum = (u32*)malloc(1 * sizeof(u32));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (impl_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    u32* d_temp;
    cudaMalloc((void**)&d_temp, grids.x/*num of blocks*/ * sizeof(u32));

    u32 shared_mem_size = BLOCK_SIZE * sizeof(u32);
    shared_reduce_sum_kernel <<< grids, blocks, shared_mem_size >>> (d_arr, impl_size, d_temp);
    // because impl_size and BLOCK_SIZE both are power of 2,
    // (impl_size / BLOCK_SIZE) is also power of 2
    shared_reduce_sum_kernel <<< 1, grids.x, grids.x * sizeof(u32) >>> (d_temp, grids.x, d_sum);
    if (cudaSuccess != cudaGetLastError()) {
        printf("shared_reduce_sum_kernel fault!\n");
    }

    cudaMemcpy(h_sum, d_sum, 1 * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = h_sum[0];
    free(h_sum);
    free(temp);
    cudaFree(d_arr);
    cudaFree(d_sum);

    return sum;
}

/**
 * use shared memory the size as the array
 * just do it once
 */
__host__
u32 gpuLargeSharedReduceSum(const u32* const h_arr, const u32 size)
{
    assert(size > 0);

    u32 impl_size = getNextPowOf2(size);
    u32* temp = (u32*)malloc(impl_size * sizeof(u32));
    COPY_ARR(temp, h_arr, size);
    ZERO_PADDING(temp, size, impl_size);
    
    u32* d_arr;
    cudaMalloc((void**)&d_arr, impl_size * sizeof(u32));
    cudaMemcpy(d_arr, temp, impl_size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_sum;
    cudaMalloc((void**)&d_sum, 1 * sizeof(u32));

    u32* h_sum = (u32*)malloc(1 * sizeof(u32));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (impl_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    u32 shared_mem_size = impl_size * sizeof(u32);
    large_shared_reduce_sum_kernel <<< grids, blocks, shared_mem_size >>> (d_arr, impl_size, d_sum);
    if (cudaSuccess != cudaGetLastError()) {
        printf("large_shared_reduce_sum_kernel fault!\n");
    }

    cudaMemcpy(h_sum, d_sum, 1 * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = *h_sum;
    free(h_sum);
    free(temp);
    cudaFree(d_arr);
    cudaFree(d_sum);

    return sum;
}

u32 gpuGlobalReduceSum(const u32* const h_arr, const u32 size)
{
    assert(size > 0);

    u32 impl_size = getNextPowOf2(size);
    u32* temp = (u32*)malloc(impl_size * sizeof(u32));
    COPY_ARR(temp, h_arr, size);
    ZERO_PADDING(temp, size, impl_size);
    
    u32* d_arr;
    cudaMalloc((void**)&d_arr, impl_size * sizeof(u32));
    cudaMemcpy(d_arr, temp, impl_size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_sum;
    cudaMalloc((void**)&d_sum, 1 * sizeof(u32));

    u32* h_sum = (u32*)malloc(1 * sizeof(u32));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (impl_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    global_reduce_sum_kernel <<< grids, blocks >>> (d_arr, impl_size, d_sum);
    if (cudaSuccess != cudaGetLastError()) {
        printf("find_max_float_kernel fault!\n");
    }

    cudaMemcpy(h_sum, d_sum, 1 * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = *h_sum;
    free(temp);
    free(h_sum);
    cudaFree(d_arr);
    cudaFree(d_sum);

    return sum;
}

u32 cpuSum(const u32* const arr, const u32 size)
{
    u32 val = 0;
    for (u32 i = 0; i < size; i++) {
        val += arr[i];
    }

    return val;
}

u32 cpuReduceSum(const u32* const arr, const u32 size)
{
    assert(size > 0);

    u32 impl_size = getNextPowOf2(size);
    u32* temp = (u32*)malloc(impl_size * sizeof(u32));
    COPY_ARR(temp, arr, size);
    ZERO_PADDING(temp, size, impl_size);

    for (u32 offset = impl_size>>1; offset > 0; offset >>= 1) {
        for (u32 i = 0; i < offset; i++) {
            temp[i] += temp[i + offset];
        }
    }
    u32 maxval = temp[0];
    free(temp);

    return maxval;
}

int main()
{
    const u32 arr_size = 1000; /* must > BLOCK_SIZE*/

    u32* arr = (u32*)malloc(arr_size * sizeof(u32));
    CREATE_RAND_ARR(arr, arr_size, 0, 10);

    u32 cpuSumVal = cpuSum(arr, arr_size);
    printf("CPU add sum: %d\n", cpuSumVal);

    u32 cpuReduceSumVal = cpuReduceSum(arr, arr_size);
    printf("CPU reduce sum: %d\n", cpuReduceSumVal);

    u32 gpuGlobalReduceSumVal = gpuGlobalReduceSum(arr, arr_size);
    printf("GPU global memory reduce sum: %d\n", gpuGlobalReduceSumVal);

    u32 gpuSharedReduceSumVal = gpuSharedReduceSum(arr, arr_size);
    printf("GPU shared memory reduce sum: %d\n", gpuSharedReduceSumVal);

    u32 gpuLargeSharedReduceSumVal = gpuLargeSharedReduceSum(arr, arr_size);
    printf("GPU large shared memory reduce sum: %d\n", gpuLargeSharedReduceSumVal);

    const u32 warp_level_size = 32; /* as same as warp size 32 */

    u32* warp_arr = (u32*)malloc(warp_level_size * sizeof(u32));
    CREATE_RAND_ARR(warp_arr, warp_level_size, 0, 10);

    u32 gpuWarpLevelReduceSumVal = gpuWarpLevelReduce(warp_arr, warp_level_size);
    printf("GPU warp level reduce sum: %d\n", gpuWarpLevelReduceSumVal);

    free(arr);
    free(warp_arr);

    return 0;
}