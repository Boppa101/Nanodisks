#ifndef GENERALFUNCTIONS_H
#define GENERALFUNCTIONS_H

#include "mkl.h"

MKL_Complex16 multiply_complex(const MKL_Complex16 z1, const MKL_Complex16 z2);
void TransposeMatMKLComplex(const MKL_INT N, MKL_Complex16* Mat);
void SortEigen(const MKL_INT N, MKL_Complex16* EVal, MKL_Complex16* EVec);
void PrintArrDouble(const int N, const int M, const double* Arr);
void PrintArrMKLComplex(const int N, const int M, const MKL_Complex16* Arr);

#endif //GENERALFUNCTIONS_H
