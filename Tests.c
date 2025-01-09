#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "/opt/intel/oneapi/mkl/2024.2/include/mkl.h"

#include "Constants.h"
#include "Functions.h"

void MatMul(int N, double** Mat, double* VecIn, double* VecOut) {
    for(int i=0; i<N; i++) {
        double res = 0;
        for(int j=0; j<N; j++) {
            res += Mat[i][j] * VecIn[j];
        }
        VecOut[i] = res;
    }
}

int main(void) {
    int N = 10;
    int m = 1;
    int cutoff = 200;
    double* ThetaArr = (double*)malloc(sizeof(double)*N);
    FillTheta(N, ThetaArr);

    double* G0Arr = (double*)malloc(sizeof(double)*N*N);

    FillM(N, m, ThetaArr, cutoff, G0Arr);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            printf("%f, ", G0Arr[i*N+j]);
        }
        printf("\n");
    }

    free(G0Arr);
    free(ThetaArr);
    return 0;
}
