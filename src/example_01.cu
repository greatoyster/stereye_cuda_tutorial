#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

int main()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: \"%s\"\n\n", dev, deviceProp.name);
        printf("Device warpSize: %d\n", deviceProp.warpSize);
        printf("Device maxGridSize: [%d, %d, %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Device maxThreadsDim: [%d, %d, %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Device maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Device multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
        printf("Device maxBlocksPerMultiProcessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    }
    return 0;
}