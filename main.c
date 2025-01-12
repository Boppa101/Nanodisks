#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include "/opt/intel/oneapi/mkl/2024.2/include/mkl.h"

#include "Constants.h"
#include "Functions.h"

void PrintArr(int N, double* Arr);
void print_matrix(char* desc, MKL_INT m, MKL_INT n, MKL_Complex16* a, MKL_INT lda);
void TransposeMat(MKL_INT N, MKL_Complex16* Mat);
void SortEigen(MKL_INT N, MKL_Complex16* EVal, MKL_Complex16* EVec);
void writeArrayToFile(const char* filename, int VecOrMat, int N, MKL_Complex16* Arr);
MKL_Complex16 multiply_complex(MKL_Complex16 z1, MKL_Complex16 z2);
void FillStrings(int N, int m, int cutoff, double EF, double omega, double gamma, double radius, char* filename_EVal, char* filename_EVec, char* filename_CD, size_t size);

int main(int argc, char** argv) {
    if(argc != 8) {
        printf("Recheck arguments!\n");
        return 1;
    }

    int N = atoi(argv[1]);
    int m = atoi(argv[2]);
    int cutoff = atoi(argv[3]);

    double EF = atof(argv[4])/au_eV;
    double omega = atof(argv[5])/au_eV;
    double gamma = atof(argv[6])/au_eV;
    double radius = atof(argv[7])/au_nm;

    printf("N=%d, m=%d, cutoff=%d, EF=%f, omega=%f, gamma=%f, radius=%f\n", N, m, cutoff, EF*au_eV, omega*au_eV, gamma*au_eV, radius*au_nm);

    MKL_Complex16 eta;
    eta.real = creal(I*Drude(EF, omega, gamma)/(omega*radius));
    eta.imag = cimag(I*Drude(EF, omega, gamma)/(omega*radius));

    char filename_EVal[512];
    char filename_EVec[512];
    char filename_CD[512];
    FillStrings(N, m, cutoff, EF, omega, gamma, radius, filename_EVal, filename_EVec, filename_CD, sizeof(filename_EVal));

    double* ThetaArr = (double*)malloc(sizeof(double)*N);
    FillTheta(N, ThetaArr);

    double* MArr = (double*)malloc(sizeof(double)*N*N);
    FillM(N, m, ThetaArr, cutoff, MArr);

    double* DArr = (double*)malloc(sizeof(double)*N*N);
    Dtilde(N, m, ThetaArr, DArr);

    // Arrays for EVals, left EVecs, right EVecs and the matrix
    MKL_INT n = N, lda = N, ldvl = N, ldvr = N, info;
    MKL_Complex16 w[N], vl[N*N], vr[N*N], a[N*N];

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            a[i*N+j].real = eta.real * MArr[i*N+j]; a[i*N+j].imag = eta.imag * MArr[i*N+j];
        }
    }

    free(ThetaArr);
    free(MArr);

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

    // Maybe extend this function so I ca specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_EVal, 0, N, w);
    writeArrayToFile(filename_EVec, 1, N, vr);

    MKL_Complex16 DArr_c[N*N];
    for(int i=0; i<N*N; i++) {
        DArr_c[i].real = DArr; DArr_c[i].imag = 0;
    }

    for(int i=0; i<10; i++) {
        // Multply DArr_c with vr[i] and save that in vr[i]
        // Basically write a function that takes a matrix and vector and saves the product in the vector
        // BE CAREFUL NOT TO OVERWRITE STUFF!!!
    }
    // To save on computation time I should not do this for all EVecs, but maybe for the first 10

    for(int i=0; i<N*N; i++) {
        MKL_Complex16 res = multiply_complex(vr[i], eta);
        vr[i].real = res.real; vr[i].imag = res.imag;
    }

    // Maybe extend this function so I ca specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_CD, 1, N, vr);

    free(DArr);
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

void writeArrayToFile(const char* filename, int VecOrMat, int N, MKL_Complex16* Arr) {
    FILE* file = fopen(filename, "w");
    if(file == NULL) {
        perror("Error opening file");
        return;
    }

    if(VecOrMat == 0) {
        for(int i=0; i<N; i++) {
            fprintf(file, "%.7f+%.7f", Arr[i].real, Arr[i].imag);
            if(i < N-1) fprintf(file, "\n");
        }
    }
    else {
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                fprintf(file, "%.7f+%.7f", Arr[i*N+j].real, Arr[i*N+j].imag);
                if(j < N-1) fprintf(file, ", ");
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}

MKL_Complex16 multiply_complex(MKL_Complex16 z1, MKL_Complex16 z2) {
    MKL_Complex16 result;
    result.real = z1.real * z2.real - z1.imag * z2.imag;
    result.imag = z1.real * z2.imag + z1.imag * z2.real;
    return result;
}

void FillStrings(
    int N, int m, int cutoff,
    double EF, double omega, double gamma, double radius,
    char* filename_EVal, char* filename_EVec, char* filename_CD,
    size_t size){
    char params[256];
    snprintf(params, sizeof(params),
             "N%d"
             "m%d"
             "cutoff%d"
             "EF%.2f"
             "omega%.2f"
             "gamma%.2f"
             "radius%.2f",
             N, m, cutoff, EF*au_eV, omega*au_eV, gamma*au_eV, radius*au_nm);

    // Replace '.' with '_'
    for (char* p = params; *p; ++p) {
        if (*p == '.') {
            *p = '_';
        }
    }

    snprintf(filename_EVal, size, "EVal_%s.txt", params);
    snprintf(filename_EVec, size, "EVec_%s.txt", params);
    snprintf(filename_CD, size, "CD___%s.txt", params);
}