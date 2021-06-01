#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>

#define u32 unsigned int
#define BLOCK_SIZE 64

#define CREATE_RAND_ARR(arr, size, min, max) \
do {                                         \
    time_t t;                                \
    srand((unsigned)time(&t));               \
    for (u32 i = 0; i < size; i++)           \
        arr[i] = rand() % max + min;         \
} while (0)                                  \

template <typename T>
T cpuFindMax(T* arr, const u32 arr_size);

template <typename T>
T gpuFindMax(T* arr, const u32 arr_size);

__device__
float atomicFloatMax(float* address, const float val)
{
    if (*address > val) {
        return *address;
    }

    int* const addressInt = (int*)address;
    int old = *addressInt, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) > val) {
            break;
        }
        old = atomicCAS(addressInt, assumed, __float_as_int(val));
    } while (old != assumed);

    return old;
}

__global__
void find_max_int_kernel(const int* __restrict__ arr, const u32 arr_size, int* maxval)
{
    __shared__ int shared_maxval;
    u32 id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x == 0) {
        shared_maxval = arr[0];
    }
    __syncthreads();
    int local_maxval = shared_maxval;
    for (u32 i = threadIdx.x; i < arr_size; i += blockDim.x) {
        local_maxval = (local_maxval > arr[i])? local_maxval : arr[i];
    }

    atomicMax(&shared_maxval, local_maxval);
    __syncthreads();

    if (id_x == 0) {
        *maxval = shared_maxval;
    }
}

__global__
void find_max_float_kernel(const float* __restrict__ arr, const u32 arr_size, float* maxval)
{
    __shared__ float shared_maxval;
    if (threadIdx.x == 0) {
        shared_maxval = arr[0];
    }
    __syncthreads();
    float local_maxval = shared_maxval;
    for (u32 i = threadIdx.x; i < arr_size; i += blockDim.x) {
        local_maxval = (local_maxval > arr[i])? local_maxval : arr[i];
    }

    atomicFloatMax(&shared_maxval, local_maxval);
    __syncthreads();

    if (threadIdx.x == 0) {
        *maxval = shared_maxval;
    }
}

template <>
int gpuFindMax<int>(int* h_arr, const u32 arr_size)
{
    assert(arr_size > 0);

    int* d_arr;
    cudaMalloc((void**)&d_arr, arr_size * sizeof(int));
    cudaMemcpy(d_arr, h_arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_maxval;
    cudaMalloc((void**)&d_maxval, 1 * sizeof(int));

    dim3 blocks = BLOCK_SIZE;
    dim3 grid = (arr_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_max_int_kernel <<< grid , blocks >>> (d_arr, arr_size, d_maxval);
    if (cudaSuccess != cudaGetLastError()) {
        printf("find_max_int_kernel fault!\n");
    }

    int* h_maxval = (int*)malloc(1 * sizeof(int));
    cudaMemcpy(h_maxval, d_maxval, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    int result = *h_maxval;

    cudaFree(d_arr);
    cudaFree(d_maxval);
    free(h_maxval);

    return result;
}

template <>
float gpuFindMax<float>(float* h_arr, const u32 arr_size)
{
    assert(arr_size > 0);

    float* d_arr;
    cudaMalloc((void**)&d_arr, arr_size * sizeof(float));
    cudaMemcpy(d_arr, h_arr, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    float* d_maxval;
    cudaMalloc((void**)&d_maxval, 1 * sizeof(float));

    dim3 blocks = BLOCK_SIZE;
    dim3 grid = (arr_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_max_float_kernel <<< grid , blocks >>> (d_arr, arr_size, d_maxval);
    if (cudaSuccess != cudaGetLastError()) {
        printf("find_max_float_kernel fault!\n");
    }

    float* h_maxval = (float*)malloc(1 * sizeof(float));
    cudaMemcpy(h_maxval, d_maxval, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float result = *h_maxval;

    cudaFree(d_arr);
    cudaFree(d_maxval);
    free(h_maxval);

    return result;
}

template <>
int cpuFindMax<int>(int* arr, const u32 arr_size)
{
    assert(arr_size > 0);
    int maxval = arr[0];
    for (u32 i = 0; i < arr_size; i++) {
        maxval = (arr[i] > maxval)? arr[i] : maxval;
    }

    return maxval;
}

template <>
float cpuFindMax<float>(float* arr, const u32 arr_size)
{
    assert(arr_size > 0);
    float maxval = arr[0];
    for (u32 i = 0; i < arr_size; i++) {
        maxval = (arr[i] > maxval)? arr[i] : maxval;
    }

    return maxval;
}

int main()
{
    const u32 arr_size = 1000;

    int* int_arr = (int*)malloc(arr_size * sizeof(int));
    CREATE_RAND_ARR(int_arr, arr_size, 0, 1000);

    float* float_arr = (float*)malloc(arr_size * sizeof(float));
    CREATE_RAND_ARR(float_arr, arr_size, 0, 1000);
    for (u32 i = 0; i < arr_size; i++) {
        float_arr[i] /= 10.0;
    }

    int cpu_int_maxval = cpuFindMax<int>(int_arr, arr_size);
    printf("CPU find max int element: %d\n", cpu_int_maxval);

    int gpu_int_maxval = gpuFindMax<int>(int_arr, arr_size);
    printf("GPU find max int element: %d\n", gpu_int_maxval);

    float cpu_float_maxval = cpuFindMax<float>(float_arr, arr_size);
    printf("CPU find max float element: %f\n", cpu_float_maxval);

    float gpu_float_maxval = gpuFindMax<float>(float_arr, arr_size);
    printf("GPU find max float element: %f\n", gpu_float_maxval);    
    
    free(int_arr);
    free(float_arr);

    return 0;
}