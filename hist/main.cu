#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>

#define u32 unsigned int
#define uchar unsigned char
#define BLOCK_SIZE 32

#define CREATE_RAND_ARR(arr, size, min, max) \
do {                                         \
    time_t t;                                \
    srand((unsigned)time(&t));               \
    for (u32 i = 0; i < size; i++)           \
        arr[i] = rand() % max + min;         \
} while (0)                                  \

#define PRINT_ARR(arr, size)       \
do {                               \
    for (u32 i = 0; i < size; i++) \
        printf("%u, ", arr[i]);    \
    printf("\n");                  \
} while (0)                        \

__device__
uchar atomicUChatAdd(uchar* address, const uchar val)
{
    unsigned long long_address_modulo = (unsigned long) address & 3;
    u32* base_address = (u32*) ((char*) address - long_address_modulo);
    u32 long_val = (u32) val << (8 * long_address_modulo);
    u32 long_old = atomicAdd(base_address, long_val);

    if (long_address_modulo == 3) {
        // the first 8 bits of long_val represent the char value,
        // hence the first 8 bits of long_old represent its previous value.
        return (char) (long_old >> 24);
    } 
    else {
        // bits that represent the char value within long_val
        unsigned int mask = 0x000000ff << (8 * long_address_modulo);
        unsigned int masked_old = long_old & mask;
        // isolate the bits that represent the char value within long_old, add the long_val to that,
        // then re-isolate by excluding bits that represent the char value
        unsigned int overflow = (masked_old + long_val) & ~mask;
        if (overflow) {
            atomicSub(base_address, overflow);
        }
        return (char) (masked_old >> 8 * long_address_modulo);
    }
}

__global__
void global_clac_hist_kernel(const uchar* __restrict__ arr, 
                             const u32 arr_size, 
                             uchar* hist, 
                             const u32 hist_bins)
{
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x < arr_size) {
        if (arr[id_x] < hist_bins) {
            atomicUChatAdd(&hist[arr[id_x]], 1);
        }
    }
}

void gpu_global_clac_hist(uchar* h_arr, const u32 arr_size, uchar* h_hist, const u32 hist_bins)
{
    uchar* d_arr;
    cudaMalloc((void**)&d_arr, arr_size * sizeof(uchar));
    cudaMemcpy(d_arr, h_arr, arr_size * sizeof(uchar), cudaMemcpyHostToDevice);

    uchar* d_hist;
    cudaMalloc((void**)&d_hist, hist_bins * sizeof(uchar));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (arr_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    global_clac_hist_kernel <<< grids, blocks >>> (d_arr, arr_size, d_hist, hist_bins);
    if (cudaSuccess != cudaGetLastError()) {
        printf("global_clac_hist_kernel fault!\n");
    }

    cudaMemcpy(h_hist, d_hist, hist_bins * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_hist);
}

void cpu_clac_hist(uchar* arr, const u32 arr_size, uchar* hist, const u32 hist_bins)
{
    for (u32 i = 0; i < arr_size; i++) {
        if (arr[i] < hist_bins) {
            hist[arr[i]]++;
        }
    }
}

int main()
{
    const u32 arr_size = 1000;
    const u32 hist_bins = 8;

    uchar* arr = (uchar*)malloc(arr_size * sizeof(uchar));
    CREATE_RAND_ARR(arr, arr_size, 0, 8);

    uchar* cpu_hist = (uchar*)malloc(hist_bins * sizeof(uchar));
    cpu_clac_hist(arr, arr_size, cpu_hist, hist_bins);
    printf("CPU clac hist:\n");
    PRINT_ARR(cpu_hist, hist_bins);

    uchar* gpu_global_hist = (uchar*)malloc(hist_bins * sizeof(uchar));
    gpu_global_clac_hist(arr, arr_size, gpu_global_hist, hist_bins);
    printf("GPU clac hist - use global memory:\n");
    PRINT_ARR(gpu_global_hist, hist_bins);

    // uchar* gpu_shared_hist = (uchar*)malloc(hist_bins * sizeof(uchar));
    // gpu_shared_clac_hist(arr, arr_size, gpu_shared_hist, hist_bins);
    // printf("GPU clac hist - use shared memory:\n");
    // PRINT_ARR(gpu_shared_hist, hist_bins);

    free(arr);
    free(cpu_hist);
    free(gpu_global_hist);
    // free(gpu_shared_hist);
    
    return 0;
}