#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "mkl.h"
#include <gsl/gsl_integration.h>

#include "Functions.h"

/*
    This file contains most of the functions used in main.c, Tests.c and
    ExtCoeff.c. Defined are potenials, charge density calculations, Frude model
    and all matrices for the calculations (D2, D1, G2, G1, G0 -> M).
    All parameters should be in the same unit system (preferably atomic units).
*/

void FillTheta(int N, double* ThetaArr) {
    for(int i=0; i<N; i++) ThetaArr[i] = 0.5*1/(double)N+(double)i/N;
}

void phi_ext(int N, const double* ThetaArr, double a, double* phi_extArr) {
    for(int i=0; i<N; i++) phi_extArr[i] = -a/2.0 * ThetaArr[i];
}

double complex Drude(double EF, double omega, double gamma) {
    return I/M_PI * EF/(omega + I*gamma);
}

void FillD1(int N, int m, double* D1Arr) {
    for(int i=0; i<N*N; i++) D1Arr[i] = 0;
    double delta = 1.0/(double)N;
    for(int i=1; i<N-1; i++) {
        D1Arr[i*N+(i-1)] = -1/(2*delta);
        D1Arr[i*N+(i+1)] = 1/(2*delta);
    }

    // Lower Boundary:
    double m_term = -pow(-1, m);
    D1Arr[0*N+0] = m_term/(2*delta);
    D1Arr[0*N+1] = 1/(2*delta);

    // Upper Boundary:
    D1Arr[(N-1)*N+(N-2)] = -1/(2*delta);
    D1Arr[(N-1)*N+(N-1)] = 1/(2*delta);
}

void FillD2(int N, int m, double* D2Arr) {
    for(int i=0; i<N*N; i++) D2Arr[i] = 0;
    double delta = 1.0/(double)N;
    double delta2 = pow(delta, 2);
    for(int i=1; i<N-1; i++) {
        D2Arr[i*N+(i-1)] = 1/delta2;
        D2Arr[i*N+i] = -2/delta2;
        D2Arr[i*N+(i+1)] = 1/delta2;
    }

    // Lower Boundary:
    double m_term = 1 - pow(-1, m);
    D2Arr[0*N+0] = (-1-m_term)/delta2;
    D2Arr[0*N+1] = 1/delta2;

    // Upper Boundary:
    D2Arr[(N-1)*N+(N-2)] = 1/delta2;
    D2Arr[(N-1)*N+(N-1)] = -1/delta2;
}

void Dtilde(int N, int m, const double* ThetaArr, double* DArr) {
    double* D1Arr = (double*)malloc(sizeof(double)*N*N);
    double* D2Arr = (double*)malloc(sizeof(double)*N*N);
    double* G0Arr = (double*)malloc(sizeof(double)*N*N);
    double* G1Arr = (double*)malloc(sizeof(double)*N*N);
    double* G1D1Arr = (double*)malloc(sizeof(double)*N*N);

    if(D1Arr==NULL || D2Arr==NULL || G0Arr==NULL || G1Arr==NULL || G1D1Arr==NULL) {
        printf("Error initialising matrices in Dtilde!\n");
        return;
    }

    FillD1(N, m, D1Arr);
    FillD2(N, m, D2Arr);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            G0Arr[i*N+j] = 0;
            G1Arr[i*N+j] = 0;
        }
        G0Arr[i*N+i] = -pow(m, 2)/pow(ThetaArr[i], 2);
        G1Arr[i*N+i] = 1/ThetaArr[i];
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, G1Arr, N, D1Arr, N, 0.0, G1D1Arr, N);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            DArr[i*N+j] = D2Arr[i*N+j] + G1D1Arr[i*N+j] + G0Arr[i*N+j];
        }
    }

    free(D1Arr);
    free(D2Arr);
    free(G0Arr);
    free(G1Arr);
    free(G1D1Arr);
}

static double* I_m0_table = NULL;
static size_t I_m0_table_size = 0;

static double* I_m1_table = NULL;
static size_t I_m1_table_size = 0;

static void load_array(const char* filename, double** array, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    double temp;
    size_t count = 0;
    while (fscanf(file, "%lf", &temp) == 1) {
        count++;
    }
    rewind(file);

    *array = (double*)malloc(count * sizeof(double));
    if (!*array) {
        perror("Error allocating memory");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < count; i++) {
        fscanf(file, "%lf", &((*array)[i]));
    }
    fclose(file);

    *size = count;
}

__attribute__((constructor)) static void initialize_arrays() {
    // printf("Initializing lookup tables...\n");
    load_array("I_m0.txt", &I_m0_table, &I_m0_table_size);
    load_array("I_m1.txt", &I_m1_table, &I_m1_table_size);
}

__attribute__((destructor)) static void cleanup_arrays() {
    // printf("Cleaning up lookup tables...\n");
    free(I_m0_table);
    free(I_m1_table);
    I_m0_table = NULL;
    I_m1_table = NULL;
    I_m0_table_size = 0;
    I_m1_table_size = 0;
}

double g_m(double x, int cutoff, int m) {
    double res = 0;
    double x_min = 0.95;
    double x_max = 0.999;

    // For values of x close to 1, a higher order is needed for accurate results
    if(x>=x_min) cutoff = 3000; // max for cutoff is 5000

    if(m==0) {
        if(x<x_max) {
            for(int l=0; l<cutoff; l++) {
                res += pow(x, 2*l) * I_m0_table[l];
            }
        }
        else {
            res = 2*M_PI - 2*log(1-pow(x, 2));
        }
    }
    else {
        if(x<x_max) {
            for(int l=0; l<cutoff; l++) {
                res += pow(x, 2*l+1) * I_m1_table[l];
            }
        }
        else {
            res = M_PI * x - 2*x*log(1 - pow(x, 2));
        }
    }
    return res;
}

typedef struct {
    int m;
    double Theta_j;
    double Theta_i;
    int cutoff;
} params_integration;

double IntegrandG2(double x, void *params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_j = p->Theta_j;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(Theta_i, x); double L = fmax(Theta_i, x);
    return 1/L * g_m(S/L, cutoff, m) * (x + x-Theta_j - pow(m, 2) * pow(x-Theta_j, 2)/(2*x));
}

double IntegrandG1(double x, void *params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_j = p->Theta_j;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(Theta_i, x); double L = fmax(Theta_i, x);
    return 1/L * g_m(S/L, cutoff, m) * (1 - pow(m, 2) * (x-Theta_j)/x);
}

double IntegrandG0(double x, void *params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    // double Theta_j = p->Theta_j;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(Theta_i, x); double L = fmax(Theta_i, x);
    return -pow(m, 2)/(L*x) * g_m(S/L, cutoff, m);
}

void FillG2(const double* ThetaArr, int m, int N, int cutoff, double* G2Arr) {
    double delta = 1/(double)N;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            params_integration params = {m, ThetaArr[j], ThetaArr[i], cutoff};
            double result1, result2, error;
            gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
            gsl_function F;
            F.function = &IntegrandG2; F.params = &params;
            gsl_integration_qags(&F, ThetaArr[j]-delta/2, ThetaArr[j], 1e-7, 1e-7, 10000, workspace, &result1, &error);
            gsl_integration_qags(&F, ThetaArr[j], ThetaArr[j]+delta/2, 1e-7, 1e-7, 10000, workspace, &result2, &error);
            gsl_integration_workspace_free(workspace);
            G2Arr[i*N+j] = result1 + result2;
        }
    }
}

void FillG1(const double* ThetaArr, int m, int N, int cutoff, double* G1Arr) {
    double delta = 1/(double)N;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            params_integration params = {m, ThetaArr[j], ThetaArr[i], cutoff};
            double result1, result2, error;
            gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
            gsl_function F;
            F.function = &IntegrandG1; F.params = &params;
            gsl_integration_qags(&F, ThetaArr[j]-delta/2, ThetaArr[j], 1e-7, 1e-7, 10000, workspace, &result1, &error);
            gsl_integration_qags(&F, ThetaArr[j], ThetaArr[j]+delta/2, 1e-7, 1e-7, 10000, workspace, &result2, &error);
            gsl_integration_workspace_free(workspace);
            G1Arr[i*N+j] = result1 + result2;
        }
    }
}

void FillG0(const double* ThetaArr, int m, int N, int cutoff, double* G0Arr) {
    double delta = 1/(double)N;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            params_integration params = {m, ThetaArr[j], ThetaArr[i], cutoff};
            double result1, result2, error;
            gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
            gsl_function F;
            F.function = &IntegrandG0; F.params = &params;
            gsl_integration_qags(&F, ThetaArr[j]-delta/2, ThetaArr[j], 1e-7, 1e-7, 10000, workspace, &result1, &error);
            gsl_integration_qags(&F, ThetaArr[j], ThetaArr[j]+delta/2, 1e-7, 1e-7, 10000, workspace, &result2, &error);
            gsl_integration_workspace_free(workspace);
            G0Arr[i*N+j] = result1 + result2;
        }
    }
}

void FillM(const int N, const int m, const double* ThetaArr, const int cutoff, double* MArr) {
    double* D1Arr = (double*)malloc(sizeof(double)*N*N);
    double* D2Arr = (double*)malloc(sizeof(double)*N*N);
    double* G0Arr = (double*)malloc(sizeof(double)*N*N);
    double* G1Arr = (double*)malloc(sizeof(double)*N*N);
    double* G2Arr = (double*)malloc(sizeof(double)*N*N);
    double* G2D2Arr = (double*)malloc(sizeof(double)*N*N);
    double* G1D1Arr = (double*)malloc(sizeof(double)*N*N);

    if(D1Arr==NULL || D2Arr==NULL || G0Arr==NULL || G1Arr==NULL || G2Arr==NULL || G2D2Arr==NULL || G1D1Arr==NULL) {
        printf("Error initialising matrices in FillM!\n");
        return;
    }

    FillD1(N, m, D1Arr);
    FillD2(N, m, D2Arr);
    FillG0(ThetaArr, m, N, cutoff, G0Arr);
    FillG1(ThetaArr, m, N, cutoff, G1Arr);
    FillG2(ThetaArr, m, N, cutoff, G2Arr);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, G2Arr, N, D2Arr, N, 0, G2D2Arr, N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, G1Arr, N, D1Arr, N, 0, G1D1Arr, N);

    for(int i=0; i<N*N; i++) MArr[i] = G2D2Arr[i] + G1D1Arr[i] + G0Arr[i];

    free(D1Arr);
    free(D2Arr);
    free(G0Arr);
    free(G1Arr);
    free(G2Arr);
    free(G2D2Arr);
    free(G1D1Arr);
}