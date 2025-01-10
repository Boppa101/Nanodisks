#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "/opt/intel/oneapi/mkl/2024.2/include/mkl.h"
#include <gsl/gsl_integration.h>

#include "Constants.h"
#include "Functions.h"

void PrintArr(int N, double* Arr);
void print_matrix(char* desc, MKL_INT m, MKL_INT n, MKL_Complex16* a, MKL_INT lda);
void TransposeMat(MKL_INT N, MKL_Complex16* Mat);
void SortEigen(MKL_INT N, MKL_Complex16* EVal, MKL_Complex16* EVec);

int main() {
    int N = 4;
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
    double radius = 25/au_nm;

    MKL_Complex16 eta;
    eta.real = creal(I*Drude(EF, omega, gamma)/(omega*radius));
    eta.imag = cimag(I*Drude(EF, omega, gamma)/(omega*radius));

    MKL_INT n = N, lda = N, ldvl = N, ldvr = N, info;

    // Arrays for EVals, left EVecs, right EVecs and the matrix
    MKL_Complex16 w[N], vl[N*N], vr[N*N], a[N*N];

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            a[i*N+j].real = eta.real * M_m0[i*N+j]; a[i*N+j].imag = eta.imag * M_m0[i*N+j];
            // a[i*N+j].real = eta.real * M_m1[i*N+j]; a[i*N+j].imag = eta.imag * M_m1[i*N+j];
        }
    }

    free(ThetaArr);
    free(M_m0);
    // free(M_m1);

    /* Solve eigenproblem */
    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, w, vl, ldvl, vr, ldvr);

    /* Check for convergence */
    if (info > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    // A given Eigevector is saved in one COLLUMN -> I transpose to have them in rows
    TransposeMat(N, vr);
    SortEigen(N, w, vr);

    /* Print eigenvalues and left and right eigenvectors */
    print_matrix("Eigenvalues", 1, n, w, 1);
    // print_matrix("Left eigenvectors", n, n, vl, ldvl);
    print_matrix("Right eigenvectors", n, n, vr, ldvr);

    // free(a);
    exit(0);
}

void TransposeMat(MKL_INT N, MKL_Complex16* Mat) {
    for(MKL_INT i=0; i<N; i++) {
        for(MKL_INT j=i+1; j<N; j++) {
            MKL_Complex16 temp;
            temp.real = Mat[i*N+j].real; temp.imag = Mat[i*N+j].imag;
            Mat[i*N+j].real = Mat[j*N+i].real; Mat[i*N+j].imag = Mat[j*N+i].imag;
            Mat[j*N+i].real = temp.real; Mat[j*N+i].imag = temp.imag;
        }
    }
}

void SortEigen(MKL_INT N, MKL_Complex16* EVal, MKL_Complex16* EVec) {
    MKL_INT i, j, k;
    for (i = 0; i < N - 1; i++) {
        for (j = 0; j < N - i - 1; j++) {
            if (EVal[j].real > EVal[j+1].real) {
                // Swap eigenvalues
                MKL_Complex16 temp_val;
                temp_val.real = EVal[j].real; temp_val.imag = EVal[j].imag;
                EVal[j].real = EVal[j+1].real; EVal[j].imag = EVal[j+1].imag;
                EVal[j+1].real = temp_val.real; EVal[j+1].imag = temp_val.imag;

                // Swap corresponding EVecs assuming one EVec is in one ROW
                for (k = 0; k < N; k++) {
                    MKL_Complex16 temp_vec;
                    temp_vec.real = EVec[j*N + k].real; temp_vec.imag = EVec[j*N + k].imag;
                    EVec[j*N + k].real = EVec[(j+1)*N + k].real; EVec[j*N + k].imag = EVec[(j+1)*N + k].imag;
                    EVec[(j+1)*N + k].real = temp_vec.real; EVec[(j+1)*N + k].imag = temp_vec.imag;
                }
            }
        }
    }
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, MKL_Complex16* a, MKL_INT lda) {
    MKL_INT i, j;
    printf("\n %s\n", desc);
    for(i=0; i<m; i++) {
        for(j=0; j<n; j++)
            printf(" (%.2f, %.2f)", a[i*lda+j].real, a[i*lda+j].imag);
        printf("\n");
    }
}

void PrintArr(int N, double* Arr) {
    for(int i=0; i<N; i++) printf("%f, ", Arr[i]);
    printf("\n");
}


// MKL_Complex16* a = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
// a[0].real = -3.84; a[0].imag =  2.25;
// a[1].real = -8.94; a[1].imag = -4.75;
// a[2].real =  8.95; a[2].imag = -6.53;
// a[3].real = -9.87; a[3].imag =  4.82;
// a[4].real = -0.66; a[4].imag =  0.83;
// a[5].real = -4.40; a[5].imag = -3.82;
// a[6].real = -3.50; a[6].imag = -4.26;
// a[7].real = -3.15; a[7].imag =  7.36;
// a[8].real = -3.99; a[8].imag = -4.73;
// a[9].real = -5.88; a[9].imag = -6.60;
// a[10].real = -3.36; a[10].imag = -0.40;
// a[11].real = -0.75; a[11].imag =  5.23;
// a[12].real =  7.74; a[12].imag =  4.18;
// a[13].real =  3.66; a[13].imag = -7.53;
// a[14].real =  2.58; a[14].imag =  3.60;
// a[15].real =  4.59; a[15].imag =  5.41;


// int main(void) {
//     // int N = 3;
//     // int cutoff = 100;
//     //
//     // double* ThetaArr = (double*)malloc(sizeof(double)*N);
//     // FillTheta(N, ThetaArr);
//     //
//     // double* M_m0 = (double*)malloc(sizeof(double)*N*N);
//     // // double* M_m1 = (double*)malloc(sizeof(double)*N*N);
//     //
//     // FillM(N, 0, ThetaArr, cutoff, M_m0);
//     // // FillM(N, 1, ThetaArr, cutoff, M_m1);
//     //
//     // double EF = 0.36/au_eV;
//     // double omega = 0.2/au_eV;
//     // double gamma = 0.02/au_eV;
//     // double a = 25/au_nm;
//     //
//     // double complex eta = I*Drude(EF, omega, gamma)/(omega*a);
//     //
//     // for(int i=0; i<N*N; i++) {
//     //     M_m0[i] = eta*M_m0[i];
//     //     // M_m1[i] = eta*M_m1[i];
//     // }
//     //
//     // double eigenvalues[N];
//     // MKL_INT n = N, lda = N, info;
//     // info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, M_m0, lda, eigenvalues);
//     //
//     // if (info > 0) {
//     //     printf("The algorithm failed to compute eigenvalues.\n");
//     //     return 1;
//     // }
//     //
//     // for (int i = 0; i < n - 1; i++) {
//     //     for (int j = 0; j < n - i - 1; j++) {
//     //         if (eigenvalues[j] > eigenvalues[j + 1]) {
//     //             // Swap eigenvalues
//     //             double temp_val = eigenvalues[j];
//     //             eigenvalues[j] = eigenvalues[j + 1];
//     //             eigenvalues[j + 1] = temp_val;
//     //
//     //             // Swap corresponding eigenvectors
//     //             for (int k = 0; k < n; k++) {
//     //                 double temp_vec = M_m0[k + j * lda];
//     //                 M_m0[k + j * lda] = M_m0[k + (j + 1) * lda];
//     //                 M_m0[k + (j + 1) * lda] = temp_vec;
//     //             }
//     //         }
//     //     }
//     // }
//     //
//     // double complex eigenvalues_c[N];
//     // for(int i=0; i<N; i++) eigenvalues_c[i] = eta*eigenvalues[i];
//     //
//     // printf("Eigenvalues:\n");
//     // for (int i = 0; i < n; i++) {
//     //     printf("%lf+%lf\n", creal(eigenvalues_c[i]), cimag(eigenvalues_c[i]));
//     // }
//     //
//     // printf("Eigenvectors (scaled):\n");
//     // for (int i = 0; i < n; i++) {
//     //     for (int j = 0; j < n; j++) {
//     //         printf("%lf ", M_m0[j + i * lda]);
//     //     }
//     //     printf("\n");
//     // }
//     //
//     // free(ThetaArr);
//     // free(M_m0);
//     // // free(M_m1);
//
//     return 0;
// }
