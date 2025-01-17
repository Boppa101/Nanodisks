#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "mkl.h"

#include "Constants.h"
#include "Functions.h"
#include "GeneralFunctions.h"

void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr);
void FillStrings(int N, int m, int cutoff, double EF, double omega, double gamma, double radius, char* filename_EVal, char* filename_EVec, char* filename_CD, size_t size);

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

    const int N         = atoi(argv[1]);
    const int m         = atoi(argv[2]);
    const int cutoff    = atoi(argv[3]);

    const double EF     = atof(argv[4])/au_eV;
    const double omega  = atof(argv[5])/au_eV;
    const double gamma  = atof(argv[6])/au_eV;
    const double radius = atof(argv[7])/au_nm;

    if(m != 0 && m != 1) {
        printf("Invalid value for m!\n");
        return 1;
    }

    printf("N=%d, m=%d, cutoff=%d, EF=%f, omega=%f, gamma=%f, radius=%f\n", N, m, cutoff, EF*au_eV, omega*au_eV, gamma*au_eV, radius*au_nm);

    const MKL_Complex16 eta = (MKL_Complex16){creal(I*Drude(EF, omega, gamma)/(omega*radius)), cimag(I*Drude(EF, omega, gamma)/(omega*radius))};

    char filename_EVal[512]; char filename_EVec[512]; char filename_CD[512];
    FillStrings(N, m, cutoff, EF, omega, gamma, radius, filename_EVal, filename_EVec, filename_CD, sizeof(filename_EVal));

    double* ThetaArr_Vec    = (double*)malloc(sizeof(double)*N);
    double* MArr_Mat        = (double*)malloc(sizeof(double)*N*N);
    double* DArr_Mat        = (double*)malloc(sizeof(double)*N*N);
    FillTheta(N, ThetaArr_Vec);
    FillM(N, m, ThetaArr_Vec, cutoff, MArr_Mat);
    Dtilde(N, m, ThetaArr_Vec, DArr_Mat);

    // Arrays for EVals, left EVecs, right EVecs and the matrix
    // vl is not needed, since I specify, that left EVecs are not to be calculated
    MKL_INT NROWS = N, NCOLS = N, NCOLSvl = N, NCOLSvr = N;
    MKL_Complex16* EVals = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N);
    MKL_Complex16* EVecsr = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
    MKL_Complex16* EPMat = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
    MKL_Complex16 EVecsl[1];

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            EPMat[i*N+j].real = eta.real * MArr_Mat[i*N+j]; EPMat[i*N+j].imag = eta.imag * MArr_Mat[i*N+j];
        }
    }

    /* Solve eigenproblem */
    MKL_INT info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', NROWS, EPMat, NCOLS, EVals, EVecsl, NCOLSvl, EVecsr, NCOLSvr);

    /* Check for convergence */
    if (info > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        return 1;
    }

    // A given Eigevector is saved in one COLLUMN -> I transpose to have them in rows
    TransposeMatMKLComplex(N, EVecsr);
    SortEigen(N, EVals, EVecsr);

    // Maybe extend this function so I ca specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_EVal, 0, N, EVals);
    writeArrayToFile(filename_EVec, 1, N, EVecsr);

    MKL_Complex16* DArr_c = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*N*N);
    for(int i=0; i<N*N; i++) {
        DArr_c[i] = (MKL_Complex16){DArr_Mat[i], 0};
    }

    const MKL_Complex16 alpha = {1.0, 0.0};
    const MKL_Complex16 beta = {0.0, 0.0};
    for(int i=0; i<N; i++) {
        MKL_Complex16 CDVec[N];
        MKL_Complex16 EVeci[N];
        for(int j=0; j<N; j++) {
            EVeci[j] = EVecsr[i*N+j];
        }
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, DArr_c, N, EVeci, 1, &beta, CDVec, 1);
        for(int j=0; j<N; j++) {
            EVecsr[i*N+j] = multiply_complex(multiply_complex(eta, (MKL_Complex16){1/radius, 0}), CDVec[j]);
        }
    }

    // Maybe extend this function so I can specify the amount of lines to write -> Never use all 100 EVecs/CD
    writeArrayToFile(filename_CD, 1, N, EVecsr);

    free(ThetaArr_Vec);
    free(MArr_Mat);
    free(DArr_Mat);
    free(EVals);
    free(EVecsr);
    free(EPMat);
    free(DArr_c);
    return 0;
}

void writeArrayToFile(const char* filename, int VecOrMat, int N, const MKL_Complex16* Arr) {
    if(Arr==NULL) {
        printf("Array cannot be saved, it is not correctly initialised!\n");
        return;
    }
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