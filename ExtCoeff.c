#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <tgmath.h>

#include "../../opt/intel/oneapi/mkl/2024.2/include/mkl.h"

#include "Constants.h"
#include "Functions.h"

MKL_Complex16 multiply_complex(const MKL_Complex16 z1, const MKL_Complex16 z2);
void print_matrix( char* desc, MKL_INT m, MKL_INT n, const MKL_Complex16* a, MKL_INT lda );

int main(int argc, char** argv) {
    const int N             = atoi(argv[1]);
    const int m             = atoi(argv[2]);
    const int cutoff        = atoi(argv[3]);

    const double EF         = atof(argv[4])/au_eV;
    const double omega_S    = atof(argv[5])/au_eV;
    const double omega_E    = atof(argv[6])/au_eV;
    const double gamma      = atof(argv[7])/au_eV;
    const double radius     = atof(argv[8])/au_nm;
    const int steps         = atoi(argv[9]);
    const double c          = 137.03599;

    double* res = (double*)malloc(sizeof(double)*N);

    double* ThetaArr = (double*)malloc(sizeof(double)*N);
    double* MArr = (double*)malloc(sizeof(double)*N*N);
    double* DArr = (double*)malloc(sizeof(double)*N*N);
    MKL_Complex16* DArr_c = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
    double* phi_extArr = (double*)malloc(sizeof(double)*N);
    MKL_Complex16* phi_extArr_c = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N);
    FillTheta(N, ThetaArr);
    FillM(N, m, ThetaArr, cutoff, MArr);
    Dtilde(N, m, ThetaArr, DArr);
    for(int i=0; i<N; i++) {
        DArr_c[i] = (MKL_Complex16){DArr[i], 0};
    }
    phi_ext(N, ThetaArr, radius, phi_extArr);

    for(int i=0; i<N; i++) {
        phi_extArr_c[i] = (MKL_Complex16){phi_extArr[i], 0};
    }

    for(int i=0; i<steps+1; i++) {
        printf("i=%d\n", i);
        const double omega = omega_S + (omega_E-omega_S)*i/steps;
        MKL_Complex16 eta;
        eta.real = creal(I*Drude(EF, omega, gamma)/(omega*radius));
        eta.imag = cimag(I*Drude(EF, omega, gamma)/(omega*radius));

        MKL_Complex16* Mtilde = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
        for(int j=0; j<N; j++) {
            for(int k=0; k<N; k++) {
                MKL_Complex16 eye = {0, 0};
                if(j==k) eye = (MKL_Complex16){1, 0};
                Mtilde[j*N+k] = (MKL_Complex16){eye.real - multiply_complex(eta, (MKL_Complex16){MArr[j*N+k], 0}).real, eye.imag - multiply_complex(eta, (MKL_Complex16){MArr[j*N+k], 0}).imag};
            }
        }

        MKL_INT nrhs = 1, lda = N, ldb = N, info;
        MKL_INT ipiv[N];
        info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, nrhs, Mtilde, lda, ipiv, phi_extArr_c, ldb);
        if (info > 0) {
            printf("The diagonal element of the triangular factor of A,\n");
            printf("U(%lld,%lld) is zero, so that A is singular;\n", info, info);
            printf("the solution could not be computed.\n");
            exit(1);
        }

        MKL_Complex16* CDArr = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N);
        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta = {0.0, 0.0};
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, DArr_c, N, phi_extArr_c, 1, &beta, CDArr, 1);
        MKL_Complex16 integral_sol = (MKL_Complex16){0, 0};
        for(int j=0; j<N; j++) {
            integral_sol.real += CDArr[j].real; integral_sol.imag += CDArr[j].imag;
        }
        integral_sol.real *= 1/(double)N; integral_sol.imag *= 1/(double)N; // Normalisation of integral

        MKL_Complex16 Polarisation = multiply_complex((MKL_Complex16){2*M_PI*radius*radius, 0}, multiply_complex(eta, integral_sol));

        double ExtCo_Area = 4*omega/(radius*radius*c) * Polarisation.imag;
        res[i] = ExtCo_Area;

        printf("%f\n", res[i]);

        free(Mtilde); // Is this necessary?
        free(CDArr); // Is this necessary?
    }

    free(ThetaArr);
    free(MArr);
    free(DArr);
    free(DArr_c);
    free(phi_extArr);
    free(phi_extArr_c);
    free(res);
    return 0;
}

MKL_Complex16 multiply_complex(const MKL_Complex16 z1, const MKL_Complex16 z2) {
    MKL_Complex16 result;
    result.real = z1.real * z2.real - z1.imag * z2.imag;
    result.imag = z1.real * z2.imag + z1.imag * z2.real;
    return result;
}

void print_matrix( char* desc, MKL_INT m, MKL_INT n, const MKL_Complex16* a, MKL_INT lda ) {
    printf("\n %s\n", desc);
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++) {
            printf(" (%6.2f,%6.2f)", a[i + j * lda].real, a[i + j * lda].imag);
        }
        printf("\n");
    }
}