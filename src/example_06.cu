#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr int N = 1000000;

int main()
{
    thrust::device_vector<float> in_dev(N);
    
    thrust::fill(in_dev.begin(), in_dev.end(), 1.0f);

    float sum = thrust::reduce(in_dev.begin(), in_dev.end());

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}