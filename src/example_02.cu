#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "example_02.h"

__global__ void kernel_vec_add(float *a, float *b, float *c, int n);

__host__ void vec_add(float *a, float *b, float *c, int n)
{
    float *dev_a, *dev_b, *dev_c;
    int size = sizeof(float) * n;

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);
    
    kernel_vec_add<<<n / 32 + 1, 32>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void kernel_vec_add(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
