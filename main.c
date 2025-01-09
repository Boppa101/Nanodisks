#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "Constants.h"
#include "Functions.h"

int main(void) {
    int N = 200;
    int cutoff = 200;

    double* ThetaArr = (double*)malloc(sizeof(double)*N*N);
    double* M_m0 = (double*)malloc(sizeof(double)*N*N);
    double* M_m1 = (double*)malloc(sizeof(double)*N*N);

    FillM(N, 0, ThetaArr, cutoff, M_m0);
    FillM(N, 1, ThetaArr, cutoff, M_m1);

    double EF = 0.36/au_eV;
    double omega = 0.2/au_eV;
    double gamma = 0.02/au_eV;
    double a = 25/au_nm;

    double complex eta = I*Drude(EF, omega, gamma);

    for(int i=0; i<N*N; i++) {
        M_m0[i] = eta*M_m0[i];
        M_m1[i] = eta*M_m1[i];
    }




    free(ThetaArr);
    free(M_m0);
    free(M_m1);
    return 0;
}
