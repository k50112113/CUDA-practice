#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cuda_kernel.cuh"

#include <iostream>
using namespace std;

#define CEIL_DIV(N, M) (N + M - 1) / M
const int max_num_threads = 1024;

__global__
void inclusive_scan_HS_kernel(float *D, int N, float *D_block) {
    int block_size   = blockDim.x;
    int i_block      = blockIdx.x;
    int block_offset = i_block * block_size;
    int n_element_this_block = (block_size < N - block_offset) ? block_size : (N - block_offset);
    int tidx = threadIdx.x;
    D += block_offset;

    for (int offset = 1 ; offset < n_element_this_block ; offset <<= 1) {
        if (tidx + offset < n_element_this_block) {
            D[tidx + offset] += D[tidx];
        }
        __syncthreads();
    }

    if (!tidx){
        D_block[i_block] = D[n_element_this_block - 1];
    }
}

__global__
void inclusive_scan_HS_kernel_correction(float *D, int N, float *D_block) {
    int block_size   = blockDim.x;
    int i_block      = blockIdx.x;
    int block_offset = i_block * block_size;
    int n_element_this_block = (block_size < N - block_offset) ? block_size : (N - block_offset);
    int tidx = threadIdx.x;
    D += block_offset;

    if (tidx < n_element_this_block && i_block > 0){
        D[tidx] += D_block[i_block - 1];
    }
}

__global__
void inclusive_scan_BL_kernel(float *D, int N, float *D_block) {
    int block_size   = 2*max_num_threads;
    int i_block      = blockIdx.x;
    int block_offset = i_block * block_size;
    int n_element_this_block = (block_size < N - block_offset) ? block_size : (N - block_offset);
    int tidx = threadIdx.x;
    D += block_offset;

    __shared__ float D_shared [2*max_num_threads];
    if (tidx < n_element_this_block) {
        D_shared[tidx] = D[tidx];
    }
    if (tidx + max_num_threads < n_element_this_block) {
        D_shared[tidx + max_num_threads] = D[tidx + max_num_threads];
    }
    __syncthreads();
    
    // sum reduction
    int offset = 1;
    for (int n_ths = n_element_this_block/2 ; n_ths > 0 ; n_ths >>= 1) {
        if (tidx < n_ths) {
            int i_element = (2*tidx + 1) * offset - 1;
            D_shared[i_element + offset] += D_shared[i_element];
        }
        offset <<= 1;
        __syncthreads();
    }
    
    float sum_tmp = D_shared[n_element_this_block - 1];
    if (!tidx) {
        D_shared[n_element_this_block - 1] = 0;
    }

    // downsweep
    offset >>= 1;
    for (int n_ths = 1 ; n_ths < n_element_this_block ; n_ths <<= 1) {
        if (tidx < n_ths) {
            int i_element = (2*tidx + 1) * offset - 1;
            float tmp = D_shared[i_element];
            D_shared[i_element]           = D_shared[i_element + offset];
            D_shared[i_element + offset] += tmp;
        }
        offset >>= 1;
        __syncthreads();
    }
    
    if (tidx + 1 < n_element_this_block) {
        D[tidx] = D_shared[tidx + 1];
    }
    if (tidx + max_num_threads + 1 < n_element_this_block) {
        D[tidx + max_num_threads] = D_shared[tidx + max_num_threads + 1];
    }
    __syncthreads();
    if (!tidx) {
        D[n_element_this_block - 1] = sum_tmp;
        D_block[i_block] = sum_tmp;
    }
}

__global__
void inclusive_scan_BL_kernel_correction(float *D, int N, float *D_block) {
    int block_size   = 2*max_num_threads;
    int i_block      = blockIdx.x;
    int block_offset = i_block * block_size;
    int n_element_this_block = (block_size < N - block_offset) ? block_size : (N - block_offset);
    int tidx = threadIdx.x;
    D += block_offset;

    if (i_block > 0) {
        if (tidx < n_element_this_block) {
            D[tidx] += D_block[i_block - 1];
        }
        if (tidx + max_num_threads < n_element_this_block) {
            D[tidx + max_num_threads] += D_block[i_block - 1];
        }
    }
}

void block_inclusive_scan(float *D_device, int N, int method) {
    // cout << "Scan each block" << endl;
    cudaError_t err = cudaGetLastError();

    int    N_block = CEIL_DIV(N, (max_num_threads * ((method == 0) ? 1 : 2)));
    float *D_block;
    cudaMalloc(&D_block, N_block * sizeof(float));
    dim3 threads(max_num_threads);
    dim3 blocks(N_block);
    if (method == 0){
        inclusive_scan_HS_kernel<<<blocks, threads>>>(D_device, N, D_block);
    }
    else if (method == 1){
        inclusive_scan_BL_kernel<<<blocks, threads>>>(D_device, N, D_block);
    }
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
    }

    // cout << "Scan the last elements of each block" << endl;

    float *dummy;
    cudaMalloc(&dummy, 1 * sizeof(float));
    dim3 threads_block(max_num_threads);
    dim3 blocks_block(1);
    if (method == 0){
        inclusive_scan_HS_kernel<<<blocks_block, threads_block>>>(D_block, N_block, dummy);
    }
    else if (method == 1){
        inclusive_scan_BL_kernel<<<blocks_block, threads_block>>>(D_block, N_block, dummy);
    }
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
    }

    // cout << "Offset each element by the scanned last elements of previous block" << endl;

    if (method == 0){
        inclusive_scan_HS_kernel_correction<<<blocks, threads>>>(D_device, N, D_block);
    }
    else if (method == 1){
        inclusive_scan_BL_kernel_correction<<<blocks, threads>>>(D_device, N, D_block);
    }
    
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
    }

}

void inclusive_scan(float *D, int N, int method){
    // int id = cudaGetDevice(&id);

    int n_padding = 0;
    if (method == 1) {
        int nn = 1;
        while (nn < N) {
            nn *= 2;
        }
        if (nn != N) {
            n_padding = nn - N;
        }
    }

    size_t D_size = N * sizeof(float);
    size_t pad_size = n_padding * sizeof(float);
    float *D_device;
    cudaMalloc(&D_device, D_size + pad_size);
    cudaMemcpy(D_device, D, D_size, cudaMemcpyHostToDevice);
    if (n_padding > 0)
        cudaMemset(D_device + N, 0, pad_size);

    block_inclusive_scan(D_device, N + n_padding, method);

    cudaMemcpy(D, D_device, D_size, cudaMemcpyDeviceToHost);

    cudaFree(D_device);
    cout << "Done" << endl;
}
