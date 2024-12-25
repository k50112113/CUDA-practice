
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "cuda_kernel.cuh"

using namespace std;

void init_rand(float *a, int n) {
    // for (int i = 0 ; i < n ; i ++) a[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0 ; i < n ; i ++) a[i] = 1.2345;
}

void init_zero(float *a, int n) {
    for (int i = 0 ; i < n ; i ++) a[i] = 0;
}

void copy(float *a, float *b, int n) {
    for (int i = 0 ; i < n ; i ++) b[i] = a[i];
}

float verify_ans(float *a, int n) {
    float tmp = 0.0;
    for (int i = 0 ; i < n ; i ++) {
        tmp += a[i];
    }
    return tmp;
}

int main(int argc, char *argv[]) {
    const int N = (1<<20);
    
    size_t D_size = N * sizeof(float);
    float *D      = (float*)malloc(D_size);
    float *D_copy = (float*)malloc(D_size);
    float res;
    init_rand(D, N);
    copy(D, D_copy, N);
    int vectorize = stoi(argv[1]);
    int verify = stoi(argv[2]);
    sum_reduction(D, N, res, vectorize);

    if (verify == 1){
        cout << "Results:      " << res << endl;
        cout << "Verification: " << verify_ans(D, N) << endl;
    }

    return 0;
}