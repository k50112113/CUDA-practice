
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "cuda_kernel.cuh"

using namespace std;

void init_rand(float *a, int n) {
    // for (int i = 0 ; i < n ; i ++) a[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0 ; i < n ; i ++) a[i] = 1.0;
}

void init_zero(float *a, int n) {
    for (int i = 0 ; i < n ; i ++) a[i] = 0;
}

void copy(float *a, float *b, int n) {
    for (int i = 0 ; i < n ; i ++) b[i] = a[i];
}

float verify_ans(float *a, float *ans, int n) {
    float tmp = 0.0;
    float l2 = 0.0;
    for (int i = 0 ; i < n ; i ++) {
        tmp += a[i];
        float l = (ans[i] - tmp);
        l2 += l * l;
    }
    return (l2 / (float)(n));
}

int main(int argc, char *argv[]) {
    float a = 1e7;
    float b = 1.0;
    float c = a + b;
    float d = b + a;

    printf("%f\n", c);
    printf("%f\n", d);
    return 0;

    const int N = (1<<16) + 3000;

    size_t D_size = N * sizeof(float);
    float *D      = (float*)malloc(D_size);
    float *D_copy = (float*)malloc(D_size);
    init_rand(D, N);
    copy(D, D_copy, N);
    // for (int i = 0 ; i < N ; i ++) cout << D[i] << " "; cout << endl;
    int method = stoi(argv[1]);
    int verify = stoi(argv[2]);
    cout << "Using method " << method << endl;
    inclusive_scan(D, N, method);
    // for (int i = 0 ; i < N ; i ++) cout << D[i] << " "; cout << endl;

    if (verify == 1){
        cout << "Verification (MSD): " << verify_ans(D_copy, D, N) << endl;
        cout << D[N-1] << endl;
    }

    return 0;
}