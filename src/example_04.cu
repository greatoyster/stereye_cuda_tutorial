#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <omp.h>

#define N (1 << 11)
#define TILE_WIDTH (1 << 4)

struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b){};

    __host__ __device__ float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);

        return dist(rng);
    }
};

__global__ void mat_mul_kernel(float *a, float *b, float *out, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n)
    {
        float val = 0;
        for (int k = 0; k < n; k++)
        {
            val += a[i * n + k] * b[k * n + j];
        }
        out[i * n + j] = val;
    }
}

__global__ void tiled_mat_mul_kernel(float *a, float *b, float *out, int n)
{
    /* Assert BlockDim.x == BlockDim.y == TILE_WIDTH */
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    const int WIDTH = n;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = bx * WIDTH + tx;
    int row = by * WIDTH + ty;

    if (row >= n || col >= n)
        return;

    float sum = 0.f;
    for (int tile_idx = 0; tile_idx < WIDTH / TILE_WIDTH; tile_idx++)
    {
        tile_A[ty][tx] = a[row * WIDTH + tile_idx * TILE_WIDTH + tx];
        tile_B[ty][tx] = b[col + (tile_idx * TILE_WIDTH + ty) * WIDTH];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }
    out[row * WIDTH + col] = sum;
}
/* C_ij=\sum_{k} A_ik*B_kj */
__host__ void mat_mul(float *a, float *b, float *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float val = 0;
            for (int k = 0; k < n; k++)
            {
                val += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = val;
        }
    }
}

__host__ void mat_mul_omp(float *a, float *b, float *out, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float val = 0;
            for (int k = 0; k < n; k++)
            {
                val += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = val;
        }
    }
}

void checktheSame(thrust::host_vector<float> &a, thrust::host_vector<float> &b)
{
    for (int i = 0; i < a.size(); i++)
    {
        if (abs(a[i] - b[i]) > 0.000001)
        {
            std::cout << "WoW" << std::endl;
        }
    }
}

void cpu_only(float *a, float *b, float *c)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    mat_mul(a, b, c, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "CPU ElapsedTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
}

void cpu_omp(float *a, float *b, float *c)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    mat_mul_omp(a, b, c, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "CPU(OMP) ElapsedTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
}

void cuda_naive(float *a, float *b, float *c)
{
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads_per_block(32, 32, 1);
    dim3 num_blocks(N / 32, N / 32, 1);

    cudaEventRecord(start);
    mat_mul_kernel<<<num_blocks, threads_per_block>>>(a, b, c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "CUDA ElapsedTime: " << elapsed_time << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void cuda_shared_memory(float *a, float *b, float *c)
{
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks(N / TILE_WIDTH, N / TILE_WIDTH, 1);

    cudaEventRecord(start);
    tiled_mat_mul_kernel<<<num_blocks, threads_per_block>>>(a, b, c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "CUDA(Tile Optimized) ElapsedTime: " << elapsed_time << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(void)
{

    std::cout << "input: " << N << "x" << N << " matrix" << std::endl;

    thrust::host_vector<float> a_host(N * N), b_host(N * N), c_host(N * N), ref_host(N * N);
    thrust::device_vector<float> a_dev(N * N), b_dev(N * N), c_dev(N * N);

    /* initialization */
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + N,
                      a_host.begin(),
                      prg(1.f, 2.f));
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + N,
                      b_host.begin(),
                      prg(1.f, 2.f));

    a_dev = a_host;
    b_dev = b_host;

    cpu_only(a_host.data(), b_host.data(), ref_host.data());

    cpu_omp(a_host.data(), b_host.data(), c_host.data());
    checktheSame(c_host, ref_host);

    cuda_naive(a_dev.data().get(), b_dev.data().get(), c_dev.data().get());
    c_host = c_dev;
    checktheSame(c_host, ref_host);

    cuda_shared_memory(a_dev.data().get(), b_dev.data().get(), c_dev.data().get());
    c_host = c_dev;
    checktheSame(c_host, ref_host);

    return 0;
}