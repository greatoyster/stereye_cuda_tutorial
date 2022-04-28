#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr int N = 1000000;

__global__ void vector_sum_kernel(float *in, float *res, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
    {
        atomicAdd(res, in[idx]);
    }
}

int main()
{
    thrust::device_vector<float> in_dev(N), sum_dev(1);

    in_dev.assign(N, 1.0f);
    sum_dev.assign(1, 0.0f);

    vector_sum_kernel<<<N / 1024 + 1, 1024>>>(in_dev.data().get(), sum_dev.data().get(), in_dev.size());

    std::cout << "Sum: " << sum_dev[0] << std::endl;

    return 0;
}