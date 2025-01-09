#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "/opt/intel/oneapi/mkl/2024.2/include/mkl.h"
#include <gsl/gsl_integration.h>

#include "Constants.h"
#include "Functions.h"

void PrintArr(int N, double* Arr) {
    for(int i=0; i<N; i++) printf("%f, ", Arr[i]);
    printf("\n");
}

// FillTheta(N, ThetaArr);
// PrintArr(N, ThetaArr);
// double complex eta = 1 + 1*I;
// for(int i=0; i<N; i++) {
//     ThetaArr[i] = ThetaArr[i] * eta;
//     printf("%f + %f\n", creal(ThetaArr[i]), cimag(ThetaArr[i]));
// }

int main(void) {
    int N = 100;
    int cutoff = 100;

    double* ThetaArr = (double*)malloc(sizeof(double)*N);
    FillTheta(N, ThetaArr);

    double* M_m0 = (double*)malloc(sizeof(double)*N*N);
    // double* M_m1 = (double*)malloc(sizeof(double)*N*N);

    FillM(N, 0, ThetaArr, cutoff, M_m0);
    // FillM(N, 1, ThetaArr, cutoff, M_m1);

    double EF = 0.36/au_eV;
    double omega = 0.2/au_eV;
    double gamma = 0.02/au_eV;
    double a = 25/au_nm;

    double complex eta = I*Drude(EF, omega, gamma)/(omega*a);

    // for(int i=0; i<N*N; i++) {
    //     M_m0[i] = eta*M_m0[i];
    //     // M_m1[i] = eta*M_m1[i];
    // }

    double eigenvalues[N];
    MKL_INT n = N, lda = N, info;
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, M_m0, lda, eigenvalues);

    if (info > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        return 1;
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (eigenvalues[j] > eigenvalues[j + 1]) {
                // Swap eigenvalues
                double temp_val = eigenvalues[j];
                eigenvalues[j] = eigenvalues[j + 1];
                eigenvalues[j + 1] = temp_val;

                // Swap corresponding eigenvectors
                for (int k = 0; k < n; k++) {
                    double temp_vec = M_m0[k + j * lda];
                    M_m0[k + j * lda] = M_m0[k + (j + 1) * lda];
                    M_m0[k + (j + 1) * lda] = temp_vec;
                }
            }
        }
    }

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%lf\n", eigenvalues[i]);
    }

    // printf("Eigenvectors:\n");
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ", creal(M_m0[j + i * lda]));
    //     }
    //     printf("\n");
    // }

    free(ThetaArr);
    free(M_m0);
    // free(M_m1);

    return 0;
}
