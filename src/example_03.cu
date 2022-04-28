#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

__global__ void vec_add_kernel(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vec_add_kernel_v2(float *a, float *b, float *c, int n)
{
    int stride = gridDim.x * blockDim.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = idx; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}
int main(void)
{
    const int N = 2 << 15;
    std::cout << "input size: " << N << std::endl;

    thrust::host_vector<float> res_host(N);
    thrust::device_vector<float> a_dev(N), b_dev(N), c_dev(N);

    thrust::fill(a_dev.begin(), a_dev.end(), 1.0f);
    thrust::fill(b_dev.begin(), b_dev.end(), 1.0f);

    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vec_add_kernel_v2<<<16*2, 1024>>>(a_dev.data().get(), b_dev.data().get(), c_dev.data().get(), c_dev.size());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "ElapsedTime: " << elapsed_time << " ms\n";

    res_host = c_dev;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}