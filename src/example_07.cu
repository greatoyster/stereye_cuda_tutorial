#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void slow_kernel(float *res)
{
    extern __shared__ float dynamic_shared_memory[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < idx; i++)
    {
        atomicAdd(res, 1);
    }
}

int main()
{
    float *sum, *another_sum;
    cudaMallocManaged(&sum, sizeof(float));
    cudaMallocManaged(&another_sum, sizeof(float));
    sum[0] = 1;
    another_sum[0] = 1;

    cudaStream_t sid;
    cudaStreamCreate(&sid);

    slow_kernel<<<16 * 20, 1024, 1 * sizeof(float)>>>(sum);
    slow_kernel<<<16 * 20, 1024, 1 * sizeof(float), sid>>>(another_sum);

    std::cout << "Sum: " << sum[0] << " (it seems our GPU is still computing" << std::endl;
    std::cout << "Another sum: " << another_sum[0] << " (it seems our GPU is still computing" << std::endl;
    std::cout << "But we can do something in CPU this moment (:" << std::endl;

    cudaDeviceSynchronize();

    std::cout << "Sum: " << sum[0] << " (it is a very slow kernel" << std::endl;
    std::cout << "Another sum: " << another_sum[0] << " (it is a very slow kernel" << std::endl;

    return 0;
}