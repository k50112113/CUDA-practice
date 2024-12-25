#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "../include/utility.h"

using namespace std;

void print_vector(float *x, int size){
    for (int i = 0 ; i < size ; i ++) cout << x[i] << " "; cout << endl;
}

void print_vector(float *x, int size, int size1){
    for (int i = 0 ; i < size ; i ++){
        for (int j = 0 ; j < size1 ; j ++) cout << x[i * size1 + j] << " "; cout << endl;
    }
}

float verify_mm(float *a, float *b, float *c, float *ans, float alpha, float beta, int M, int K, int N){
    float l2 = 0.0;
    for (int m = 0 ; m < M ; m ++){
        for (int n = 0 ; n < N ; n ++){
            float tmp = 0.0;
            for (int k = 0 ; k < K ; k ++){
                tmp += a[m * K + k] * b[k * N + n];
            }
            tmp = ans[m * N + n] - (alpha * tmp + beta * c[m * N + n]);
            l2 += tmp * tmp;
        }
    }
    return l2 / (float)(M*N);
}