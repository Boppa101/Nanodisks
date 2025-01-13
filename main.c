#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <tgmath.h>

#include "../../opt/intel/oneapi/mkl/2024.2/include/mkl.h"

#include "Constants.h"
#include "Functions.h"

void PrintArr(int N, const double* Arr);
void print_matrix(char* desc, MKL_INT m, MKL_INT n, const MKL_Complex16* a, MKL_INT lda);
void TransposeMat(MKL_INT N, MKL_Complex16* Mat);
void SortEigen(MKL_INT N, MKL_Complex16* EVal, MKL_Complex16* EVec);
void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr);
MKL_Complex16 multiply_complex(MKL_Complex16 z1, MKL_Complex16 z2);
void FillStrings(int N, int m, int cutoff, double EF, double omega, double gamma, double radius, char* filename_EVal, char* filename_EVec, char* filename_CD, size_t size);
int TestParams(int m, double EF, double omega);

int main(const int argc, char** argv) {
    if(argc==1) {
        printf("This program calculates the eigenvalues and -vectors of a nanodisc for a given set of parameters.\n");
        printf("\nParameters:\nN:\tSize of Arrays and Matrices (NxN)\nm:\tAzimuthal Symmetrie\n"
               "cutoff:\tCutoff for g_m\nEF:\tFermi level in eV\nomega:\tOptical frequency in eV\n"
               "gamma:\tDamping rate in ev\nradius:\tDiskradius in nm\n\n");
        printf("Eigenvalues, -vectors and charge densities calculated from eigenvectors are saved in files:\n"
               "EVal_N(Val)m(Val)cutoff(Val)EF(Val)omega(Val)gamma(Val)radius(Val).txt\n"
               "EVec_N(Val)m(Val)cutoff(Val)EF(Val)omega(Val)gamma(Val)radius(Val).txt\n"
               "CD___N(Val)m(Val)cutoff(Val)EF(Val)omega(Val)gamma(Val)radius(Val).txt\n");
        return 1;
    }
    if(argc != 8) {
        printf("Recheck arguments!\n");
        return 1;
    }

    const int N = atoi(argv[1]);
    const int m = atoi(argv[2]);
    const int cutoff = atoi(argv[3]);

    const double EF = atof(argv[4])/au_eV;
    const double omega = atof(argv[5])/au_eV;
    const double gamma = atof(argv[6])/au_eV;
    const double radius = atof(argv[7])/au_nm;

    const int TestParams_int = TestParams(m, EF, omega);
    if(TestParams_int == 1) {
        return 1;
    }

    printf("N=%d, m=%d, cutoff=%d, EF=%f, omega=%f, gamma=%f, radius=%f\n", N, m, cutoff, EF*au_eV, omega*au_eV, gamma*au_eV, radius*au_nm);

    MKL_Complex16 eta;
    eta.real = creal(I*Drude(EF, omega, gamma)/(omega*radius));
    eta.imag = cimag(I*Drude(EF, omega, gamma)/(omega*radius));

    char filename_EVal[512]; char filename_EVec[512]; char filename_CD[512];
    FillStrings(N, m, cutoff, EF, omega, gamma, radius, filename_EVal, filename_EVec, filename_CD, sizeof(filename_EVal));

    double ThetaArr[N];
    double MArr[N*N];
    double DArr[N*N];
    FillTheta(N, ThetaArr);
    FillM(N, m, ThetaArr, cutoff, MArr);
    Dtilde(N, m, ThetaArr, DArr);

    // Arrays for EVals, left EVecs, right EVecs and the matrix
    // vl is not needed, since I specify, that left EVecs are not to be calculated
    MKL_INT NROWS = N, NCOLS = N, NCOLSvl = N, NCOLSvr = N, info;
    MKL_Complex16 EVals[N], EVecsl[1], EVecsr[N*N], EPMat[N*N];

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            EPMat[i*N+j].real = eta.real * MArr[i*N+j]; EPMat[i*N+j].imag = eta.imag * MArr[i*N+j];
        }
    }

    /* Solve eigenproblem */
    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', NROWS, EPMat, NCOLS, EVals, EVecsl, NCOLSvl, EVecsr, NCOLSvr);

    /* Check for convergence */
    if (info > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    // A given Eigevector is saved in one COLLUMN -> I transpose to have them in rows
    TransposeMat(N, EVecsr);
    SortEigen(N, EVals, EVecsr);

    // Maybe extend this function so I ca specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_EVal, 0, N, EVals);
    writeArrayToFile(filename_EVec, 1, N, EVecsr);

    MKL_Complex16 DArr_c[N*N];
    for(int i=0; i<N*N; i++) {
        DArr_c[i].real = DArr[i]; DArr_c[i].imag = 0;
    }

    MKL_Complex16 EVeci[N];
    MKL_Complex16 CDVec[N];
    MKL_Complex16 alpha = {1.0, 0.0};
    MKL_Complex16 beta = {0.0, 0.0};
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            EVeci[i] = EVecsr[i*N+j];
        }
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, DArr_c, N, EVeci, 1, &beta, CDVec, 1);
        for(int j=0; j<N; j++) {
            EVecsr[i*N+j] = CDVec[j];
        }
    }

    for(int i=0; i<N*N; i++) {
        MKL_Complex16 res = multiply_complex(EVecsr[i], eta);
        EVecsr[i].real = res.real; EVecsr[i].imag = res.imag;
    }

    // Maybe extend this function so I ca specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_CD, 1, N, EVecsr);

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
    for(MKL_INT i=0; i<N-1; i++) {
        for(MKL_INT j=0; j<N-i-1; j++) {
            if(EVal[j].real > EVal[j+1].real) {
                // Swap eigenvalues
                MKL_Complex16 temp_val;
                temp_val.real = EVal[j].real; temp_val.imag = EVal[j].imag;
                EVal[j].real = EVal[j+1].real; EVal[j].imag = EVal[j+1].imag;
                EVal[j+1].real = temp_val.real; EVal[j+1].imag = temp_val.imag;

                // Swap corresponding EVecs assuming one EVec is in one ROW
                for(MKL_INT k=0; k<N; k++) {
                    MKL_Complex16 temp_vec;
                    temp_vec.real = EVec[j*N + k].real; temp_vec.imag = EVec[j*N + k].imag;
                    EVec[j*N + k].real = EVec[(j+1)*N + k].real; EVec[j*N + k].imag = EVec[(j+1)*N + k].imag;
                    EVec[(j+1)*N + k].real = temp_vec.real; EVec[(j+1)*N + k].imag = temp_vec.imag;
                }
            }
        }
    }
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, const MKL_Complex16* a, MKL_INT lda) {
    printf("\n %s\n", desc);
    for(MKL_INT i=0; i<m; i++) {
        for(MKL_INT j=0; j<n; j++)
            printf(" (%.2f, %.2f)", a[i*lda+j].real, a[i*lda+j].imag);
        printf("\n");
    }
}

void PrintArr(int N, const double* Arr) {
    for(int i=0; i<N; i++) printf("%f, ", Arr[i]);
    printf("\n");
}

void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr) {
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

    snprintf(filename_EVal, size, "../Data/EVal_%s.txt", params);
    snprintf(filename_EVec, size, "../Data/EVec_%s.txt", params);
    snprintf(filename_CD, size, "../Data/CD___%s.txt", params);
}

int TestParams(int m, double EF, double omega) {
    if(m!=0 && m!=1) {
        printf("Invalid value for m!\n");
        return 1;
    }
    if(EF < omega) {
        printf("Warning: Implemented is Drude conductivity and expects EF>omega!\n");
    }
    return 0;
}