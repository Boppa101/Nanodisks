#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

// #include "../../opt/intel/oneapi/mkl/2024.2/include/mkl.h"
#include <math.h>

#include "mkl.h"

#include "Constants.h"
#include "Functions.h"

#include <time.h>

MKL_Complex16 multiply_complex(const MKL_Complex16 z1, const MKL_Complex16 z2);
void print_matrix( char* desc, MKL_INT m, MKL_INT n, const MKL_Complex16* a, MKL_INT lda );
void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr);

// MKL_Complex16 random_complex() {
//     MKL_Complex16 z;
//     z.real = (rand() % 200 - 100) / 10.0;  // Real part in range [-10, 10)
//     z.imag = (rand() % 200 - 100) / 10.0;  // Imaginary part in range [-10, 10)
//     return z;
// }
//
// int main() {
//     const int N = 10;
//     long long int* ipiv = (long long int*)malloc(sizeof(long long int)*N);
//
//     // Seed the random number generator
//     srand((unsigned int)time(NULL));
//
//     // Allocate memory for A, B, and ipiv
//     MKL_Complex16 *A = (MKL_Complex16 *)mkl_malloc(N * N * sizeof(MKL_Complex16), 64);
//     MKL_Complex16 *B = (MKL_Complex16 *)mkl_malloc(N * sizeof(MKL_Complex16), 64);
//
//     if (A == NULL || B == NULL || ipiv == NULL) {
//         printf("Memory allocation failed.\n");
//         free(A);
//         free(B);
//         free(ipiv);
//         return 1;
//     }
//
//     // Generate random complex matrix A and vector B
//     for (int i = 0; i < N * N; i++) {
//         A[i] = random_complex();
//     }
//     for (int i = 0; i < N; i++) {
//         B[i] = random_complex();
//     }
//
//     // Solve the system A * X = B
//     int info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, 1, A, N, ipiv, B, 1); // ldb was N before, when COL_MAJOR
//
//     // Check the result
//     if (info == 0) {
//         printf("\nSolution X:\n");
//         for (int i = 0; i < N; i++) {
//             printf("(%5.2f, %5.2f)\n", B[i].real, B[i].imag);
//         }
//     } else if (info < 0) {
//         printf("Error: Parameter %d had an illegal value.\n", -info);
//     } else {
//         printf("Error: Matrix A is singular.\n");
//     }
//
//     // Free allocated memory
//     mkl_free(A);
//     mkl_free(B);
//     mkl_free(ipiv);
//
//     printf("DONE!\n");
//     return 0;
// }


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

    printf("N=%d, m=%d, cutoff=%d, EF=%f, omega_S=%f, omega_E=%f, gamma=%f, radius=%f, steps=%d\n", N, m, cutoff, EF*au_eV, omega_S*au_eV, omega_E*au_eV, gamma*au_eV, radius*au_nm, steps);

    double* Theta_Vec               = (double*)mkl_malloc(sizeof(double)*N, 64);
    double* M_Mat                   = (double*)mkl_malloc(sizeof(double)*N*N, 64);
    double* D_Mat                   = (double*)mkl_malloc(sizeof(double)*N*N, 64);
    MKL_Complex16* D_Mat_c          = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N*N, 64);
    double* phi_ext_Vec             = (double*)mkl_malloc(sizeof(double)*N, 64);
    MKL_Complex16* phi_ext_Vec_c    = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N, 64);
    double* res_Vec                 = (double*)mkl_malloc(sizeof(double)*(steps+1), 64);
    MKL_Complex16* Mtilde_Mat       = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N*N, 64);
    MKL_Complex16* CD_Vec           = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N, 64);
    MKL_INT* ipiv                   = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*N, 64);

    FillTheta(N, Theta_Vec);
    FillM(N, m, Theta_Vec, cutoff, M_Mat);
    Dtilde(N, m, Theta_Vec, D_Mat);
    for(int i=0; i<N*N; i++) { D_Mat_c[i] = (MKL_Complex16){D_Mat[i], 0}; }
    phi_ext(N, Theta_Vec, radius, phi_ext_Vec);

    for(int i=0; i<steps+1; i++) {
        for(int j=0; j<N; j++) { phi_ext_Vec_c[j] = (MKL_Complex16){phi_ext_Vec[j], 0}; }
        // for(int f=0; f<N; f++) {
        //     printf("%f+%f, ", phi_ext_Vec_c[f].real, phi_ext_Vec_c[f].imag);
        // }
        // Correct
        // printf("\n");
        const double omega = omega_S + (omega_E-omega_S)*((double)i)/((double)steps);
        MKL_Complex16 eta;
        eta.real = creal(I*Drude(EF, omega, gamma)/(omega*radius));
        eta.imag = cimag(I*Drude(EF, omega, gamma)/(omega*radius));
        // printf("%f+%f\n", eta.real, eta.imag); //Correct

        for(int j=0; j<N; j++) {
            for(int k=0; k<N; k++) {
                double eye = 0.0;
                if(j==k) eye = 1.0;
                Mtilde_Mat[j*N+k] = (MKL_Complex16){eye - multiply_complex(eta, (MKL_Complex16){M_Mat[j*N+k], 0}).real, - multiply_complex(eta, (MKL_Complex16){M_Mat[j*N+k], 0}).imag};
            }
        }
        // printf("%f+%f, %f+%f\n", Mtilde_Mat[0].real, Mtilde_Mat[0].imag, Mtilde_Mat[1].real, Mtilde_Mat[1].imag); // Looks good

        long long int nrhs = 1, lda = N, ldb = 1, info; //, ldb = N
        info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, nrhs, Mtilde_Mat, lda, ipiv, phi_ext_Vec_c, ldb);
        if (info > 0) {
            printf("The diagonal element of the triangular factor of A,\n");
            printf("U(%lld,%lld) is zero, so that A is singular;\n", info, info);
            printf("the solution could not be computed.\n");
            exit(1);
        }
        if(info < 0) printf("Error: Parameter %lld had an illegal value.\n", -info);
        // if(i==0) {
        //     for(int f=0; f<N; f++) {
        //         printf("%f+%f, ", phi_ext_Vec_c[f].real, phi_ext_Vec_c[f].imag);
        //     }
        //     printf("\n");
        // }

        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta = {0.0, 0.0};
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, D_Mat_c, N, phi_ext_Vec_c, 1, &beta, CD_Vec, 1);

        // if(i==0) {
        //     for(int f=0; f<N; f++) {
        //         printf("%f+%f, ", CD_Vec[f].real, CD_Vec[f].imag);
        //     }
        //     printf("\n");
        // }

        MKL_Complex16 integral_sol = (MKL_Complex16){0, 0};
        for(int j=0; j<N; j++) {
            // if(j==0) {
            //     printf("%f+%f, %f+%f\n", CD_Vec[0].real * Theta_Vec[0] * Theta_Vec[0], CD_Vec[0].imag * Theta_Vec[0] * Theta_Vec[0], CD_Vec[1].real * Theta_Vec[1] * Theta_Vec[1], CD_Vec[1].imag * Theta_Vec[1] * Theta_Vec[1]);
            // }
            integral_sol.real += CD_Vec[j].real * Theta_Vec[j] * Theta_Vec[j]; integral_sol.imag += CD_Vec[j].imag * Theta_Vec[j] * Theta_Vec[j];
        }
        integral_sol.real *= 1.0/(double)N; integral_sol.imag *= 1.0/(double)N; // THIS IS NOT A GOOD ENOUGH APPROX?!?!?!

        MKL_Complex16 Polarisation = multiply_complex((MKL_Complex16){2*M_PI*radius*radius, 0}, multiply_complex(eta, integral_sol));

        double ExtCo_Area = 4.0*omega/(radius*radius*c) * Polarisation.imag;
        res_Vec[i] = ExtCo_Area;
        printf("%f, %f\n", omega*au_eV, res_Vec[i]);
    }

    // writeArrayToFile(filename, VecOrMat, N, Arr);

    mkl_free(Theta_Vec);
    mkl_free(M_Mat);
    mkl_free(D_Mat);
    mkl_free(D_Mat_c);
    mkl_free(phi_ext_Vec);
    mkl_free(phi_ext_Vec_c);
    mkl_free(res_Vec);
    mkl_free(Mtilde_Mat);
    mkl_free(CD_Vec);
    mkl_free(ipiv);
    exit(0);
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

void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr) {
    FILE* file = fopen(filename, "w");
    if(file == NULL) {
        perror("Error opening file");
        return;
    }

    if(VecOrMat == 0) {
        for(int i=0; i<N; i++) {
            fprintf(file, "%.10f+%.10f", Arr[i].real, Arr[i].imag);
            if(i < N-1) fprintf(file, "\n");
        }
    }
    else {
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                fprintf(file, "%.10f+%.10f", Arr[i*N+j].real, Arr[i*N+j].imag);
                if(j < N-1) fprintf(file, ", ");
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}