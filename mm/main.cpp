
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "include/cuda_kernel.cuh"
#include "include/utility.h"

using namespace std;

void init_rand(float *a, int n){
    for (int i = 0 ; i < n ; i ++) a[i] = (float)rand() / (float)RAND_MAX;
}

void init_zero(float *a, int n){
    for (int i = 0 ; i < n ; i ++) a[i] = 0;
}

int main(int argc, char *argv[]) {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    // const int M = 4096;
    // const int K = 4096;
    // const int N = 4096;
    
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);
    float *a = (float*)malloc(a_size);
    float *b = (float*)malloc(b_size);
    float *c = (float*)malloc(c_size);
    float *c_copied = (float*)malloc(c_size);
    init_rand(a, M * K);
    init_rand(b, K * N);
    init_rand(c, M * N);
    for (int i = 0 ; i < M * N ; i ++) c_copied[i] = c[i];
    float alpha = (float)rand() / (float)RAND_MAX;
    float beta  = (float)rand() / (float)RAND_MAX;

    int method = stoi(argv[1]);
    int verify = stoi(argv[2]);
    cout << "Using method " << method << endl;
    matmul(a, b, c, alpha, beta, M, K, N, method);

    if (verify == 1){
        cout << "Verification (MSD): " << verify_mm(a, b, c_copied, c, alpha, beta, M, K, N) << endl;
    }

    return 0;
}