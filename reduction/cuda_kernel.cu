// https://godbolt.org/z/jvofa5dhj

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cuda_kernel.cuh"

#include <iostream>
using namespace std;

#define CEIL_DIV(N, M) (N + M - 1) / M
#define WARPSIZE 32
#define FULLMASK 0xffffffff

__global__
void sum_reduction_kernel_no_vectorize(float *D, int N, int n_element_per_block, float *res) {
    __shared__ float warp_shuffle_tmp [WARPSIZE];

    const int num_threads  = blockDim.x;
    const int i_block      = blockIdx.x;
    const int block_offset = i_block * n_element_per_block;
    const int n_element_this_block = (n_element_per_block < N - block_offset) ? n_element_per_block : (N - block_offset);
    D += block_offset;
    int tidx = threadIdx.x;

    float sub_total = 0.0;

    for (int thread_offset = 0 ; thread_offset < n_element_this_block ; thread_offset += num_threads) {

        if (tidx + thread_offset < n_element_this_block) {
            sub_total += D[tidx + thread_offset];
            // printf("%d %f %f %f %f | %d\n", tidx, vec.x, vec.y, vec.z, vec.w, n_element_this_block);
        }

    }

    __syncthreads();

    const int warp_id = tidx / WARPSIZE;
    const int warp_lane = tidx % WARPSIZE;

    # pragma unroll
    for (int offset = WARPSIZE / 2 ; offset > 0 ; offset >>= 1) {
        sub_total += __shfl_down_sync(FULLMASK, sub_total, offset);
    }
    if (!warp_lane) {
        warp_shuffle_tmp[warp_id] = sub_total;
    }
    __syncthreads();
    if (!warp_id) {
        sub_total = warp_shuffle_tmp[warp_lane];
        # pragma unroll
        for (int offset = WARPSIZE / 2 ; offset > 0 ; offset >>= 1) {
            sub_total += __shfl_down_sync(FULLMASK, sub_total, offset);
        }
        if (!tidx) {
            atomicAdd(&res[0], sub_total);
        }
    }

}
// Welford (normalization)
// CBU
__global__
void sum_reduction_kernel(float *D, int N, int n_element_per_block, float *res) {
    __shared__ float warp_shuffle_tmp [WARPSIZE];

    const int num_threads  = blockDim.x;
    const int i_block      = blockIdx.x;
    const int block_offset = i_block * n_element_per_block;
    const int n_element_this_block = (n_element_per_block < N - block_offset) ? n_element_per_block : (N - block_offset);
    D += block_offset;
    int tidx = threadIdx.x;

    float sub_total = 0.0;

    for (int thread_offset = 0 ; thread_offset < n_element_this_block ; thread_offset += num_threads * 4) {

        if (tidx * 4 + thread_offset < n_element_this_block) {
            float4 vec = reinterpret_cast<float4*> (&D[tidx * 4 + thread_offset])[0];
            sub_total += vec.x;
            sub_total += vec.y;
            sub_total += vec.z;
            sub_total += vec.w;
            // sub_total += D[tidx * 4 + thread_offset + 0];
            // sub_total += D[tidx * 4 + thread_offset + 1];
            // sub_total += D[tidx * 4 + thread_offset + 2];
            // sub_total += D[tidx * 4 + thread_offset + 3];
            // printf("%d %f %f %f %f | %d\n", tidx, vec.x, vec.y, vec.z, vec.w, n_element_this_block);
        }

    }

    __syncthreads();

    const int warp_id = tidx / WARPSIZE;
    const int warp_lane = tidx % WARPSIZE;

    # pragma unroll
    for (int offset = WARPSIZE / 2 ; offset > 0 ; offset >>= 1) {
        sub_total += __shfl_down_sync(FULLMASK, sub_total, offset);
    }
    if (!warp_lane) {
        warp_shuffle_tmp[warp_id] = sub_total;
    }
    __syncthreads();
    if (!warp_id) {
        sub_total = warp_shuffle_tmp[warp_lane];
        # pragma unroll
        for (int offset = WARPSIZE / 2 ; offset > 0 ; offset >>= 1) {
            sub_total += __shfl_down_sync(FULLMASK, sub_total, offset);
        }
        if (!tidx) {
            atomicAdd(&res[0], sub_total);
        }
    }

}

void sum_reduction(float *D, int N, float &res, int vectorize){
    // int id = cudaGetDevice(&id);

    int n_padding = (4 - (N % 4)) % 4;
    const int max_num_threads = 1024;

    size_t D_size = N * sizeof(float);
    size_t pad_size = n_padding * sizeof(float);
    float *D_device;
    cudaMalloc(&D_device, D_size + pad_size);
    cudaMemcpy(D_device, D, D_size, cudaMemcpyHostToDevice);
    if (n_padding > 0)
        cudaMemset(D_device + N, 0x0, pad_size);
    float *res_device;
    cudaMalloc(&res_device, 1 * sizeof(float));
    cudaMemset(res_device, 0, 1 * sizeof(float));

    int N_padded = N + n_padding;
    
    if (vectorize == 1) {
        cout << "Use vectorize" << endl;
        int n_threads_per_block = max_num_threads / 4;
        int n_element_per_block = max_num_threads;
        // int n_element_per_block = N_padded;
        int n_blocks = CEIL_DIV(N_padded, n_element_per_block);
        dim3 threads(n_threads_per_block);
        dim3 blocks(n_blocks);
        cout << "Number of threads per block: " << n_threads_per_block << endl;
        cout << "Number of elements per block: " << n_element_per_block << endl;
        cout << "Number of blocks: " << n_blocks << endl;
        sum_reduction_kernel<<<blocks, threads>>>(D_device, N_padded, n_element_per_block, res_device);
    }
    else  {
        cout << "Close vectorize" << endl;
        int n_threads_per_block = max_num_threads / 4;
        int n_element_per_block = max_num_threads;
        // int n_element_per_block = N_padded;
        int n_blocks = CEIL_DIV(N_padded, n_element_per_block);
        dim3 threads(n_threads_per_block);
        dim3 blocks(n_blocks);
        cout << "Number of threads per block: " << n_threads_per_block << endl;
        cout << "Number of elements per block: " << n_element_per_block << endl;
        cout << "Number of blocks: " << n_blocks << endl;
        sum_reduction_kernel_no_vectorize<<<blocks, threads>>>(D_device, N_padded, n_element_per_block, res_device);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
    }

    cudaMemcpy(&res, res_device, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(D_device);
    cudaFree(res_device);
    cout << "Done" << endl;
}
