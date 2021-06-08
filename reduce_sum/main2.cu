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

__inline__ __device__
u32 warpReduceSum(u32 val) {
    for (u32 offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__inline__ __device__
u32 blockReduceSum(u32 val) 
{
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction
    if (lane==0) shared[wid]=val; // Write reduced value to shared memory
    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ 
void deviceReduceBlockAtomicKernel(u32* in, u32* out, u32 N) 
{
    u32 sum = 0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x) {
      sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
      atomicAdd(out, sum);
}

u32 cpuSum(const u32* const arr, const u32 size)
{
    u32 val = 0;
    for (u32 i = 0; i < size; i++) {
        val += arr[i];
    }

    return val;
}

int main()
{
    const u32 size = 2000;
    u32* in = (u32*)malloc(size * sizeof(u32));
    u32* out = (u32*)malloc(1 * sizeof(u32));

    CREATE_RAND_ARR(in, size, 0, 100);

    u32* d_in;
    cudaMalloc((void**)&d_in, size * sizeof(u32));
    cudaMemcpy(d_in, in, size * sizeof(u32), cudaMemcpyHostToDevice);

    u32* d_out;
    cudaMalloc((void**)&d_out, 1 * sizeof(u32));

    dim3 blocks = BLOCK_SIZE;
    dim3 grids = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    deviceReduceBlockAtomicKernel <<< grids, blocks >>> (d_in, d_out, size);
    if (cudaSuccess != cudaGetLastError()) {
        printf("deviceReduceBlockAtomicKernel fault!\n");
    }

    cudaMemcpy(out, d_out, 1 * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = cpuSum(in, size);

    printf("CPU SUM: %u\n", sum);
    printf("GPU SUM: %u\n", *out);

    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}