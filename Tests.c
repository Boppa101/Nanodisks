#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "/opt/intel/oneapi/mkl/2024.2/include/mkl.h"
#include <gsl/gsl_integration.h>

#include "Constants.h"
#include "Functions.h"

// This file contains multiple tests for the functions defined in Functions.c.
// Tested are the "actions" of the matrices included in M and the Coulomb potential.
// We can see an error in Gtildetilde for the case of m=1.
// N:            Number of points in the theta array
// m:            Azimuthal mode number in exp(i m \varphi)
// cutoff:       Cutoff order in Fun.g_m(x, cutoff, m)
// EF:           Fermi energy in eV
// gamma:        Damping factor in eV
// a:            Radius of the nanodisk in nm
// wStart:       Lower frequency bound in eV
// wEnd:         Upper frequency bound in eV

void PrintArr(int N, double* Arr);
int TestD1(int N, int m, double* ThetaArr);
int TestD2(int N, int m, double* ThetaArr);
int TestDTilde(int N, int m, double* ThetaArr);
void TestCoulomb(int N, int m, int cutoff, double* ThetaArr, double* res);
int TestG2(int N, int m, int cutoff, double* ThetaArr);
int TestG1(int N, int m, int cutoff, double* ThetaArr);
int TestG0(int N, int m, int cutoff, double* ThetaArr);
int TestM(int N, int m, int cutoff, double* ThetaArr);

int main(void) {
    int N = 100;
    int m = 0;
    int cutoff = 100;
    double* ThetaArr = (double*)malloc(sizeof(double)*N);
    FillTheta(N, ThetaArr);

    int res = TestM(N, m, cutoff, ThetaArr);
    // int res = TestDTilde(N, m, ThetaArr);
    // int res = TestG0(N, m, cutoff, ThetaArr);
    printf("%d\n", res);

    // double* res = (double*)malloc(sizeof(double)*N);
    // TestCoulomb(N, m, cutoff, ThetaArr, res);
    // PrintArr(N, res);
    // free(res);

    free(ThetaArr);
    return 0;
}

double phi(double x, int m) {
    if(m==0)    return cos(M_PI*x);
    else        return sin(M_PI/2*x);
}

double dphi(double x, int m) {
    if(m==0)    return -M_PI*sin(M_PI*x);
    else        return M_PI/2*cos(M_PI/2*x);
}

double ddphi(double x, int m) {
    if(m==0)    return -pow(M_PI, 2)*cos(M_PI*x);
    else        return -pow(M_PI/2, 2)*sin(M_PI/2*x);
}

int TestD1(int N, int m, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    double* dy = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);
    for(int i=0; i<N; i++) dy[i] = dphi(ThetaArr[i], m);

    double* D1Arr = (double*)malloc(sizeof(double)*N*N);
    FillD1(N, m, D1Arr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, D1Arr, N, y, 1, 0.0, res, 1);

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - dy[i]) >= 1e-3) Flag_res = 1;
    }

    free(y);
    free(dy);
    free(D1Arr);
    free(res);

    return Flag_res;
}

int TestD2(int N, int m, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    double* ddy = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);
    for(int i=0; i<N; i++) ddy[i] = ddphi(ThetaArr[i], m);

    double* D2Arr = (double*)malloc(sizeof(double)*N*N);
    FillD2(N, m, D2Arr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, D2Arr, N, y, 1, 0.0, res, 1);

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - ddy[i]) >= 1e-3) Flag_res = 1;
    }

    free(y);
    free(ddy);
    free(D2Arr);
    free(res);

    return Flag_res;
}

int TestDTilde(int N, int m, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    double* Dy = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);
    for(int i=0; i<N; i++) Dy[i] = ddphi(ThetaArr[i], m) + 1/ThetaArr[i]*dphi(ThetaArr[i], m) - pow(m, 2)/pow(ThetaArr[i], 2)*phi(ThetaArr[i], m);
    // if(m==0) for(int i=0; i<N; i++) Dy[i] = -pow(M_PI, 2)*cos(M_PI*ThetaArr[i]) - 1/ThetaArr[i]*M_PI*sin(M_PI*ThetaArr[i]);
    // else for(int i=0; i<N; i++) Dy[i] = -pow(M_PI/2, 2)*sin(M_PI/2*ThetaArr[i]) + 1/ThetaArr[i]*M_PI/2*cos(M_PI/2*ThetaArr[i]) - pow(m, 2)/pow(ThetaArr[i], 2)*sin(M_PI/2*ThetaArr[i]);

    double* DtildeArr = (double*)malloc(sizeof(double)*N*N);
    Dtilde(N, m, ThetaArr, DtildeArr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, DtildeArr, N, y, 1, 0.0, res, 1);

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - Dy[i]) >= 2e-2) Flag_res = 1;
    }

    free(y);
    free(Dy);
    free(DtildeArr);
    free(res);

    return Flag_res;
}

void TestCoulomb(int N, int m, int cutoff, double* ThetaArr, double* res) {
    for(int i=0; i<N; i++) res[i] = g_m(ThetaArr[i], cutoff, m);
}

typedef struct {
    int m;
    double Theta_i;
    int cutoff;
} params_integration;

double IntegrandG2_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return x/L * g_m(S/L, cutoff, m) * phi(x, m);
}

int TestG2(int N, int m, int cutoff, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);

    double* G2Arr = (double*)malloc(sizeof(double)*N*N);
    FillG2(ThetaArr, m, N, cutoff, G2Arr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, G2Arr, N, y, 1, 0.0, res, 1);

    double* res_ana = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) {
        params_integration params = {m, ThetaArr[i], cutoff};
        double result1, result2, error;
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
        gsl_function F;
        F.function = &IntegrandG2_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result1, &error);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-7, 10000, workspace, &result2, &error);
        res_ana[i] = result1 + result2;
        gsl_integration_workspace_free(workspace);
    }

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - res_ana[i]) >= 2e-2) Flag_res = 1;
    }

    free(y);
    free(G2Arr);
    free(res);
    free(res_ana);
    return Flag_res;
}

double IntegrandG1_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return 1/L * g_m(S/L, cutoff, m) * phi(x, m);
}

int TestG1(int N, int m, int cutoff, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);

    double* G1Arr = (double*)malloc(sizeof(double)*N*N);
    FillG1(ThetaArr, m, N, cutoff, G1Arr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, G1Arr, N, y, 1, 0.0, res, 1);

    double* res_ana = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) {
        params_integration params = {m, ThetaArr[i], cutoff};
        double result1, result2, error;
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
        gsl_function F;
        F.function = &IntegrandG1_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result1, &error);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-5, 10000, workspace, &result2, &error);
        res_ana[i] = result1 + result2;
        gsl_integration_workspace_free(workspace);
    }

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - res_ana[i]) >= 2e-2) Flag_res = 1;
    }

    free(y);
    free(G1Arr);
    free(res);
    free(res_ana);
    return Flag_res;
}

double IntegrandG0_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return -pow(m, 2)/(L*x) * g_m(S/L, cutoff, m) * phi(x, m);
}

// This test will fail for m=1 -> related to divergence at Theta=0
int TestG0(int N, int m, int cutoff, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);

    double* G0Arr = (double*)malloc(sizeof(double)*N*N);
    FillG0(ThetaArr, m, N, cutoff, G0Arr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, G0Arr, N, y, 1, 0.0, res, 1);

    double* res_ana = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) {
        params_integration params = {m, ThetaArr[i], cutoff};
        double result1, result2, error;
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
        gsl_function F;
        F.function = &IntegrandG0_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result1, &error);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-5, 10000, workspace, &result2, &error);
        res_ana[i] = result1 + result2;
        gsl_integration_workspace_free(workspace);
    }

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - res_ana[i]) >= 2e-2) Flag_res = 1;
    }

    free(y);
    free(G0Arr);
    free(res);
    free(res_ana);
    return Flag_res;
}

double IntegrandM2_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return x/L * g_m(S/L, cutoff, m) * ddphi(x, m);
}

double IntegrandM1_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return 1/L * g_m(S/L, cutoff, m) * dphi(x, m);
}

double IntegrandM0_Test(double x, void* params) {
    params_integration *p = (params_integration *)params;
    int m = p->m;
    double Theta_i = p->Theta_i;
    int cutoff = p->cutoff;
    double S = fmin(x, Theta_i);
    double L = fmax(x, Theta_i);
    return -pow(m, 2)/(L*x) * g_m(S/L, cutoff, m) * phi(x, m);
}

int TestM(int N, int m, int cutoff, double* ThetaArr) {
    double* y = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) y[i] = phi(ThetaArr[i], m);

    double* MArr = (double*)malloc(sizeof(double)*N*N);
    FillM(N, m, ThetaArr, cutoff, MArr);

    double* res = (double*)malloc(sizeof(double)*N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, 1.0, MArr, N, y, 1, 0.0, res, 1);

    double* res_ana = (double*)malloc(sizeof(double)*N);
    for(int i=0; i<N; i++) {
        params_integration params = {m, ThetaArr[i], cutoff};
        double result1, result2, error1;
        double result3, result4, error2;
        double result5, result6, error3;
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(10000);
        gsl_function F;
        F.function = &IntegrandM2_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result1, &error1);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-5, 10000, workspace, &result2, &error1);
        F.function = &IntegrandM1_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result3, &error2);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-5, 10000, workspace, &result4, &error2);
        F.function = &IntegrandM0_Test; F.params = &params;
        gsl_integration_qags(&F, 0, ThetaArr[i], 1e-7, 1e-7, 10000, workspace, &result5, &error3);
        gsl_integration_qags(&F, ThetaArr[i], 1, 1e-7, 1e-5, 10000, workspace, &result6, &error3);
        res_ana[i] = result1 + result2 + result3 + result4 + result5 + result6;
        gsl_integration_workspace_free(workspace);
    }

    int Flag_res = 0;
    for(int i=0; i<N; i++) {
        if(fabs(res[i] - res_ana[i]) >= 1e-2) Flag_res = 1;
    }

    free(y);
    free(MArr);
    free(res);
    free(res_ana);
    return Flag_res;
}

void PrintArr(int N, double* Arr) {
    for(int i=0; i<N; i++) printf("%f, ", Arr[i]);
    printf("\n");
}