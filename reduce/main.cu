#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>

#define u32 unsigned int
#define u64 unsigned long
#define uchar unsigned char
#define BLOCK_SIZE 64

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

u32 cpuSum(const u32* const arr, const u32 size)
{
    u32 maxval = 0;
    for (u32 i = 0; i < size; i++) {
        maxval += arr[i];
    }

    return maxval;
}

u32 cpuReduceClacSum(const u32* const arr, const u32 size)
{
    assert(size > 0);

    u32 impl_size = 1;
    if (!IS_POWER_OF_2(size)) {
        while (impl_size < size) {
            impl_size <<= 1;
        }
    }
    else {
        impl_size = size;
    }
    u32* temp = (u32*)malloc(impl_size * sizeof(u32));
    COPY_ARR(temp, arr, size);
    ZERO_PADDING(temp, size, impl_size);

    for (u32 offset = impl_size/2; offset > 0; offset /= 2) {
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
    const u32 arr_size = 1000;

    u32* arr = (u32*)malloc(arr_size * sizeof(u32));
    CREATE_RAND_ARR(arr, arr_size, 0, 10);

    u32 cpuSumval = cpuSum(arr, arr_size);
    printf("CPU add sum: %d\n", cpuSumval);

    u32 cpuReduceSumval = cpuReduceClacSum(arr, arr_size);
    printf("CPU reduce sum: %d\n", cpuReduceSumval);

    free(arr);
    return 0;
}