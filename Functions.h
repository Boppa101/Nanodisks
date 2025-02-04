#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <complex.h>

void phi_ext(int N, const double* ThetaArr, double a, double* phi_extArr);
void FillTheta(int N, double* ThetaArr);
double complex Drude(double EF, double omega, double gamma);

void FillD1(int N, int m, double* D1Arr);
void FillD2(int N, int m, double* D2Arr);
void Dtilde(int N, int m, const double* ThetaArr, double* DArr);

double g_m(double x, int cutoff, int m);

void FillG0(const double* ThetaArr, int m, int N, int cutoff, double* G0Arr);
void FillG1(const double* ThetaArr, int m, int N, int cutoff, double* G1Arr);
void FillG2(const double* ThetaArr, int m, int N, int cutoff, double* G2Arr);
void FillM(int N, int m, const double* ThetaArr, int cutoff, double* MArr);

#endif //FUNCTIONS_H
