#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/cuda_kernel.cuh"

#include <iostream>
using namespace std;

// a (M, K), b (K, N), c (M, N)

#define ceil_div(N, M) (N + M - 1) / M
#define BLOCKSIZE 32
#define SHARED_MEM_SIZE (BLOCKSIZE * BLOCKSIZE)∂
const int WARPSIZE = 32;

__global__
void matmul_baseline(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N){
    int m = BLOCKSIZE * blockIdx.x + threadIdx.x;
    int n = BLOCKSIZE * blockIdx.y + threadIdx.y;
    if (m < M && n < N){
        float tmp = 0.0;
        for(int k = 0 ; k < K ; k ++){
            tmp += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = alpha * tmp + beta * c[m * N + n];
    }
}

__global__
void matmul_gmem_coalescing(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N){
    int m = BLOCKSIZE * blockIdx.x + (threadIdx.x / BLOCKSIZE);
    int n = BLOCKSIZE * blockIdx.y + (threadIdx.x % BLOCKSIZE);
    if (m < M && n < N){
        float tmp = 0.0;
        for(int k = 0 ; k < K ; k++){
            tmp += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = alpha * tmp + beta * c[m * N + n];
    }
}

__global__ 
void matmul_smem_tiling(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N){
    __shared__ float A[SHARED_MEM_SIZE];
    __shared__ float B[SHARED_MEM_SIZE];

    const int blockm = blockIdx.x;
    const int blockn = blockIdx.y;
    const int threadm = threadIdx.x / BLOCKSIZE;
    const int threadn = threadIdx.x % BLOCKSIZE;

    float* a_block = a + blockm * BLOCKSIZE * K;
    float* b_block = b + blockn * BLOCKSIZE;
    float* c_block = c + blockm * BLOCKSIZE * N + blockn * BLOCKSIZE;

    float tmp = 0.0;
    for(int i_block = 0 ; i_block < K ; i_block += BLOCKSIZE){
        
        A[threadm * BLOCKSIZE + threadn] = a_block[threadm * K + threadn];
        B[threadm * BLOCKSIZE + threadn] = b_block[threadm * N + threadn];

        __syncthreads();

        a_block += BLOCKSIZE;
        b_block += BLOCKSIZE * N;

        for (int k = 0; k < BLOCKSIZE ; k++) {
            tmp += A[threadm * BLOCKSIZE + k] * B[k * BLOCKSIZE + threadn];
        }
        
        __syncthreads();

    }
    c_block[threadm * N + threadn] = alpha * tmp + beta * c_block[threadm * N + threadn];

}

template <const int BM, const int BN, const int BK, const int TM>
__global__ 
void matmul_smem_1D_blocktiling(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N) {
    __shared__ float A[BM * BK];
    __shared__ float B[BK * BN];

    const int blockm = blockIdx.x;
    const int blockn = blockIdx.y;
    const int threadm = threadIdx.x / BN;
    const int threadn = threadIdx.x % BN;

    float* a_block = a + blockm * BM * K;
    float* b_block = b + blockn * BN;
    float* c_block = c + blockm * BM * N + blockn * BN;

    // assert(BM * BK == blockDim.x);
    // assert(BN * BK == blockDim.x);
    // This means the number of entries A and B have in shared memory should be equal to the number of threads, so that one all threads span through A and B. 
    // This leads to BK * TM = BN = BM. make sure BM, BN, BK, and TM satisfy this relation.
    const int threadm_A = threadIdx.x / BK;
    const int threadn_A = threadIdx.x % BK;
    const int threadm_B = threadIdx.x / BN;
    const int threadn_B = threadIdx.x % BN;

    float c_reg[TM] = {0.0};

    for (int k_block = 0 ; k_block < K ; k_block += BK) {

        A[threadm_A * BK + threadn_A] = a_block[threadm_A * K + threadn_A];
        B[threadm_B * BN + threadn_B] = b_block[threadm_B * N + threadn_B];

        __syncthreads();

        a_block += BK;
        b_block += BK * N;

        for (int k_BK = 0 ; k_BK < BK ; k_BK ++) {
            float B_reg = B[k_BK * BN + threadn];

            for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
                c_reg[k_TM] += A[(threadm * TM + k_TM) * BK + k_BK] * B_reg;
            }
        }

        __syncthreads();
    }

    for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
        c_block[(threadm * TM + k_TM) * N + threadn] = alpha * c_reg[k_TM] + beta * c_block[(threadm * TM + k_TM) * N + threadn];
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void 
matmul_smem_2D_blocktiling(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N) {
    __shared__ float A[BM * BK];
    __shared__ float B[BK * BN];

    const int blockm = blockIdx.x;
    const int blockn = blockIdx.y;
    const int threadm = threadIdx.x / (BN / TN);
    const int threadn = threadIdx.x % (BN / TN);

    float* a_block = a + blockm * BM * K;
    float* b_block = b + blockn * BN;
    float* c_block = c + blockm * BM * N + blockn * BN;

    const int threadm_A = threadIdx.x / BK;
    const int threadn_A = threadIdx.x % BK;
    const int stride_A  = blockDim.x  / BK;
    const int threadm_B = threadIdx.x / BN;
    const int threadn_B = threadIdx.x % BN;
    const int stride_B  = blockDim.x  / BN;

    float A_reg[TM] = {0.0};
    float B_reg[TN] = {0.0};
    float c_reg[TM * TN] = {0.0};

    for (int k_block = 0 ; k_block < K ; k_block += BK) {
        
        for (int offsetm_A = 0 ; offsetm_A < BM ; offsetm_A += stride_A) {
            A[(offsetm_A + threadm_A) * BK + threadn_A] = a_block[(offsetm_A + threadm_A) * K + threadn_A];
        }
        // transpose A (but seems slower)
        // for (int offsetm_A = 0 ; offsetm_A < BM ; offsetm_A += stride_A) {
        //     A[threadn_A * BM + offsetm_A + threadm_A] = a_block[(offsetm_A + threadm_A) * K + threadn_A];
        // }
        for (int offsetk_B = 0 ; offsetk_B < BK ; offsetk_B += stride_B) {
            B[(offsetk_B + threadm_B) * BN + threadn_B] = b_block[(offsetk_B + threadm_B) * N + threadn_B];
        }
        __syncthreads();

        a_block += BK;
        b_block += BK * N;

        for (int k_BK = 0 ; k_BK < BK ; k_BK ++) {
            
            for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
                A_reg[k_TM] = A[(threadm * TM + k_TM) * BK + k_BK];
                // transpose A (but seems slower)
                // A_reg[k_TM] = A[k_BK * BM + threadm * TM + k_TM];
            }
            for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                B_reg[k_TN] = B[k_BK * BN + threadn * TN + k_TN];
            }

            for (int k_TM = 0 ; k_TM < TM ; k_TM ++ ) {
                for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                    c_reg[k_TM * TN + k_TN] += A_reg[k_TM] * B_reg[k_TN];
                }
            }
        }

        __syncthreads();
    }
    
    for (int k_TM = 0 ; k_TM < TM ; k_TM ++ ) {
        for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
            c_block[(threadm * TM + k_TM) * N + threadn * TN + k_TN] = alpha * c_reg[k_TM * TN + k_TN] + beta * c_block[(threadm * TM + k_TM) * N + threadn * TN + k_TN];
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ 
void matmul_gmem_vectorized(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N) {
    __shared__ float A[BM * BK];
    __shared__ float B[BK * BN];

    const int blockm = blockIdx.x;
    const int blockn = blockIdx.y;
    const int threadm = threadIdx.x / (BN / TN);
    const int threadn = threadIdx.x % (BN / TN);

    float* a_block = a + blockm * BM * K;
    float* b_block = b + blockn * BN;
    float* c_block = c + blockm * BM * N + blockn * BN;

    const int threadm_A = threadIdx.x / (BK / 4);
    const int threadn_A = threadIdx.x % (BK / 4);
    const int threadm_B = threadIdx.x / (BN / 4);
    const int threadn_B = threadIdx.x % (BN / 4);

    float A_reg[TM]      = {0.0};
    float B_reg[TN]      = {0.0};
    float c_reg[TM * TN] = {0.0};

    for (int k_block = 0 ; k_block < K ; k_block += BK) {

        reinterpret_cast<float4*> (&A[threadm_A * BK + threadn_A * 4])[0] = reinterpret_cast<float4*> (&a_block[threadm_A * K + threadn_A * 4])[0];

        // transpose A
        // float4 a_tmp = reinterpret_cast<float4*>(&a_block[threadm_A * K + threadn_A * 4])[0];
        // A[(threadn_A * 4 + 0) * BM + threadm_A] = a_tmp.x;
        // A[(threadn_A * 4 + 1) * BM + threadm_A] = a_tmp.y;
        // A[(threadn_A * 4 + 2) * BM + threadm_A] = a_tmp.z;
        // A[(threadn_A * 4 + 3) * BM + threadm_A] = a_tmp.w;

        reinterpret_cast<float4*> (&B[threadm_B * BN + threadn_B * 4])[0] = reinterpret_cast<float4*> (&b_block[threadm_B * N + threadn_B * 4])[0];
        
        // transpose B
        // float4 b_tmp = reinterpret_cast<float4*>(&b_block[threadm_B * N + threadn_B * 4])[0];
        // B[(threadn_B * 4 + 0) * BK + threadm_B] = b_tmp.x;
        // B[(threadn_B * 4 + 1) * BK + threadm_B] = b_tmp.y;
        // B[(threadn_B * 4 + 2) * BK + threadm_B] = b_tmp.z;
        // B[(threadn_B * 4 + 3) * BK + threadm_B] = b_tmp.w;

        __syncthreads();

        a_block += BK;
        b_block += BK * N;

        for (int k_BK = 0 ; k_BK < BK ; k_BK ++) {
            
            for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
                A_reg[k_TM] = A[(threadm * TM + k_TM) * BK + k_BK];
                // A_reg[k_TM] = A[k_BK * BM + threadm * TM + k_TM];
            }
            for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                B_reg[k_TN] = B[k_BK * BN + threadn * TN + k_TN];
                // B_reg[k_TN] = B[(threadn * TN + k_TN) * BK + k_BK];
            }

            for (int k_TM = 0 ; k_TM < TM ; k_TM ++ ) {
                for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                    c_reg[k_TM * TN + k_TN] += A_reg[k_TM] * B_reg[k_TN];
                }
            }
        }

        __syncthreads();
    }
    
    for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
        for (int k_TN = 0 ; k_TN < TN ; k_TN += 4) {
            float4 c_tmp = reinterpret_cast<float4*> (&c_block[(threadm * TM + k_TM) * N + threadn * TN + k_TN])[0];
            c_tmp.x = alpha * c_reg[k_TM * TN + k_TN + 0] + beta * c_tmp.x;
            c_tmp.y = alpha * c_reg[k_TM * TN + k_TN + 1] + beta * c_tmp.y;
            c_tmp.z = alpha * c_reg[k_TM * TN + k_TN + 2] + beta * c_tmp.z;
            c_tmp.w = alpha * c_reg[k_TM * TN + k_TN + 3] + beta * c_tmp.w;
            reinterpret_cast<float4*> (&c_block[(threadm * TM + k_TM) * N + threadn * TN + k_TN])[0] = c_tmp;
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNitr, const int TM, const int TN, const int NUM_THREADS>
__global__ 
void __launch_bounds__ (NUM_THREADS) matmul_warptiling(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N) {
    __shared__ float A[BM * BK];
    __shared__ float B[BK * BN];

    const int blockm      = blockIdx.x;
    const int blockn      = blockIdx.y;
    const int warpidx     = threadIdx.x / WARPSIZE;
    const int warpm       = warpidx / (BN / WN);
    const int warpn       = warpidx % (BN / WN);
    constexpr int WMitr   = (WM * WN) / (WARPSIZE * TM * TN) / WNitr;
    constexpr int WsubM   = WM / WMitr;
    constexpr int WsubN   = WN / WNitr;
    const int threadidx_inwarp = threadIdx.x % WARPSIZE;
    const int threadm_inwarp = threadidx_inwarp / (WsubN / TN);
    const int threadn_inwarp = threadidx_inwarp % (WsubN / TN);

    float* a_block = a + blockm * BM * K;
    float* b_block = b + blockn * BN;
    float* c_block = c + (blockm * BM + warpm * WM) * N + blockn * BN + warpn * WN;

    const int threadm_A = threadIdx.x / (BK / 4);
    const int threadn_A = threadIdx.x % (BK / 4);
    constexpr int stridem_A = NUM_THREADS / (BK / 4) ;
    const int threadm_B = threadIdx.x / (BN / 4);
    const int threadn_B = threadIdx.x % (BN / 4);
    constexpr int stridem_B = NUM_THREADS / (BN / 4) ;

    float A_reg[WMitr * TM]              = {0.0};
    float B_reg[WNitr * TN]              = {0.0};
    float c_reg[WMitr * TM * WNitr * TN] = {0.0};

    for (int k_block = 0 ; k_block < K ; k_block += BK) {
        
        for (int offsetm = 0 ; (offsetm + stridem_A) <= BM ; offsetm += stridem_A) {
            // reinterpret_cast<float4*> (&A[(threadm_A + offsetm) * BK + threadn_A * 4])[0] = reinterpret_cast<float4*> (&a_block[(threadm_A + offsetm) * K + threadn_A * 4])[0];
            
            // transpose A
            float4 a_tmp = reinterpret_cast<float4*>(&a_block[(threadm_A + offsetm) * K + threadn_A * 4])[0];
            A[(threadn_A * 4 + 0) * BM + (threadm_A + offsetm)] = a_tmp.x;
            A[(threadn_A * 4 + 1) * BM + (threadm_A + offsetm)] = a_tmp.y;
            A[(threadn_A * 4 + 2) * BM + (threadm_A + offsetm)] = a_tmp.z;
            A[(threadn_A * 4 + 3) * BM + (threadm_A + offsetm)] = a_tmp.w;

        }

        
        for (int offsetm = 0 ; (offsetm + stridem_B) <= BK ; offsetm += stridem_B) {
            reinterpret_cast<float4*> (&B[(threadm_B + offsetm) * BN + threadn_B * 4])[0] = reinterpret_cast<float4*> (&b_block[(threadm_B + offsetm) * N + threadn_B * 4])[0];
        }

        __syncthreads();

        a_block += BK;
        b_block += BK * N;

        for (int k_BK = 0 ; k_BK < BK ; k_BK ++) {
            
            for (int k_WM = 0 ; k_WM < WMitr ; k_WM ++) {
                for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
                    // A_reg[k_WM * TM + k_TM] = A[(warpm * WM + k_WM * WsubM + threadm_inwarp * TM + k_TM) * BK + k_BK];
                    A_reg[k_WM * TM + k_TM] = A[k_BK * BM + warpm * WM + k_WM * WsubM + threadm_inwarp * TM + k_TM];
                }
            }

            for (int k_WN = 0 ; k_WN < WNitr ; k_WN ++) {
                for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                    B_reg[k_WN * TN + k_TN] = B[k_BK * BN + warpn * WN + k_WN * WsubN + threadn_inwarp * TN + k_TN];
                }
            }

            for (int k_WM = 0 ; k_WM < WMitr ; k_WM ++) {
                for (int k_WN = 0 ; k_WN < WNitr ; k_WN ++) {
                    for (int k_TM = 0 ; k_TM < TM ; k_TM ++ ) {
                        for (int k_TN = 0 ; k_TN < TN ; k_TN ++) {
                            c_reg[(k_WM * TM + k_TM) * (WNitr * TN) + k_WN * TN + k_TN] += A_reg[k_WM * TM + k_TM] * B_reg[k_WN * TN + k_TN];
                        }
                    }
                }
            }

        }

        __syncthreads();
    }
    
    
    for (int k_WM = 0 ; k_WM < WMitr ; k_WM ++) {
        for (int k_WN = 0 ; k_WN < WNitr ; k_WN ++) {
            float* c_warp_tmp = c_block + (k_WM * WsubM) * N + k_WN * WsubM;
            for (int k_TM = 0 ; k_TM < TM ; k_TM ++) {
                for (int k_TN = 0 ; k_TN < TN ; k_TN += 4) {
                    float4 c_tmp = reinterpret_cast<float4*> (&c_warp_tmp[(threadm_inwarp * TM + k_TM) * N + threadn_inwarp * TN + k_TN])[0];
                    const int c_idx = (k_WM * TM + k_TM) * (WNitr * TN) + k_WN * TN + k_TN;
                    c_tmp.x = alpha * c_reg[c_idx + 0] + beta * c_tmp.x;
                    c_tmp.y = alpha * c_reg[c_idx + 1] + beta * c_tmp.y;
                    c_tmp.z = alpha * c_reg[c_idx + 2] + beta * c_tmp.z;
                    c_tmp.w = alpha * c_reg[c_idx + 3] + beta * c_tmp.w;
                    reinterpret_cast<float4*> (&c_warp_tmp[(threadm_inwarp * TM + k_TM) * N + threadn_inwarp * TN + k_TN])[0] = c_tmp;
                }
            }
        }
    }

}

void matmul(float *a, float *b, float *c, float alpha, float beta, int M, int K, int N, int method){
    // int id = cudaGetDevice(&id);

    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);
    float* a_device;
    float* b_device;
    float* c_device;
    cudaMalloc(&a_device, a_size);
    cudaMalloc(&b_device, b_size);
    cudaMalloc(&c_device, c_size);
    cudaMemcpy(a_device, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c, c_size, cudaMemcpyHostToDevice);

    // cudaMallocManaged(&a_device, bytes);
    // cudaMallocManaged(&b_device, bytes);
    // cudaMallocManaged(&c_device, bytes);
    // cudaMemPrefetchAsync(a_device, bytes, id);
    // cudaMemPrefetchAsync(b_device, bytes, id);

    // threads -> blockDim
    // size_t shared_memory_size = SHARED_MEM_SIZE * sizeof(float);
    cublasHandle_t handle;
    if (method == -1) {
        cublasCreate(&handle); 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_device, N, a_device, K, &beta, c_device, N);
    }
    else if (method == 0) {
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        dim3 blocks(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
        matmul_baseline<<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 1) {
        dim3 threads(BLOCKSIZE * BLOCKSIZE);
        dim3 blocks(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
        matmul_gmem_coalescing<<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 2) { 
        dim3 threads(BLOCKSIZE * BLOCKSIZE);
        dim3 blocks(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
        matmul_smem_tiling<<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 3) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 8;
        const uint TM = 8;
        // BM = BN = BK * TM
        dim3 threads((BM * BN) / TM);
        dim3 blocks(ceil_div(M, BM), ceil_div(N, BN));
        matmul_smem_1D_blocktiling<BM, BN, BK, TM><<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 4) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 16;
        const uint TM = 4;
        const uint TN = 4;
        dim3 threads((BM * BN) / (TM * TN));
        dim3 blocks(ceil_div(M, BM), ceil_div(N, BN));
        matmul_smem_2D_blocktiling<BM, BN, BK, TM, TN><<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 5) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 16;
        const uint TM = 4;
        const uint TN = 4;
        // BM = BN = (BK * TM * TN) / 4 -> all threads span shared mem
        dim3 threads((BM * BN) / (TM * TN));
        dim3 blocks(ceil_div(M, BM), ceil_div(N, BN));
        matmul_gmem_vectorized<BM, BN, BK, TM, TN><<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    } else if (method == 6) {
        
        // BM * BN = Number of entries a block process per BK segment per kernel launch
        const uint BM = 32; // BM = 32 (16 blocks/SM, 128 threads/block) and BM = 64 (8 blocks/SM, 256 threads/block) both have 2048 threads/SM, which is the upper limit of number of threads per SM, but the former is faster.
        const uint BN = 64;
        const uint BK = 16;
        // WM * WN = Number of entries a warp process per block
        const uint WM = 16;
        const uint WN = 32;
        // TM * TN = Number of entries a thread process per in-warp-thread-iteration
        const uint TM = 4;
        const uint TN = 4;
        // WMitr * WNitr = (WM * WN) / (WARPSIZE * TM * TN) = Number of iterations a warp undergoes, each iteration is a in-warp-thread-iteration
        const uint WNitr = 1;
        constexpr uint NUM_THREADS = (BM * BN) / (WM * WN) * WARPSIZE;

        // warptile in threadblocktile
        static_assert((BN % WN == 0) and (BM % WM == 0));
        static_assert((BN / WN) * (BM / WM) == NUM_THREADS / WARPSIZE);

        // threads in warpsubtile
        static_assert((WM * WN) % (WARPSIZE * TM * TN * WNitr) ==
                      0);
        constexpr uint WMitr = (WM * WN) / (WARPSIZE * TM * TN * WNitr);
        constexpr uint WsubM = WM / WMitr;
        constexpr uint WsubN = WN / WNitr;
        // warpsubtile in warptile
        static_assert((WM % WMitr == 0) and (WN % WNitr == 0));

        static_assert((NUM_THREADS * 4) % BK == 0,
                      "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                      "issues during GMEM->SMEM tiling (loading only parts of the "
                      "final row of Bs during each iteraion)");
        static_assert((NUM_THREADS * 4) % BN == 0,
                      "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                      "issues during GMEM->SMEM tiling (loading only parts of the "
                      "final row of As during each iteration)");
        static_assert(WsubN % TN == 0,∂
                      "WsubN must be a multiple of TN to avoid quantization effects");
        static_assert(WsubM % TM == 0,
                      "WsubM must be a multiple of TM to avoid quantization effects");
        static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                      "BM*BK must be a multiple of 4*NUM_THREADS to vectorize loads");
        static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                      "BN*BK must be a multiple of 4*NUM_THREADS to vectorize loads");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t max_shared_mem_per_block = prop.sharedMemPerBlock;
        size_t max_shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
        size_t shared_mem = (BM + BN) * BK * sizeof(float);
        int    max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        cout << "Shared memory: " << (BM + BN) * BK * sizeof(float) << "/(" << max_shared_mem_per_block << ", " << max_shared_mem_per_sm << ") (Number of Blocks/SM: " << max_shared_mem_per_sm/shared_mem << ")" << endl;
        cout << "Threads: " << NUM_THREADS << "/" << max_threads_per_sm << " (Number of Blocks/SM: " << max_threads_per_sm/NUM_THREADS << ")" << endl;
        cout << "BM, BN, BK: " << BM << ", " << BN << ", " << BK << endl;
        cout << "WM, WN: " << WM << ", " << WN << endl;
        cout << "WMitr, WNitr: " << WMitr << ", " << WNitr << endl;
        cout << "WsubM, WsubN: " << WsubM << ", " << WsubN << endl;
        cout << "TM, TN: " << TM << ", " << TN << endl;

        dim3 threads(NUM_THREADS);
        dim3 blocks(ceil_div(M, BM), ceil_div(N, BN));
        matmul_warptiling<BM, BN, BK, WM, WN, WNitr, TM, TN, NUM_THREADS><<<blocks, threads>>>(a_device, b_device, c_device, alpha, beta, M, K, N);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(c, c_device, c_size, cudaMemcpyDeviceToHost);
    // cudaMemPrefetchAsync(d_c, bytes, cudaCpuDeviceId);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    if (method == -1) {
        cublasDestroy(handle);
    }
    cout << "Done" << endl;
}
