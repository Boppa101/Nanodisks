#include <stdio.h>

#include "mkl.h"

#include "GeneralFunctions.h"

MKL_Complex16 multiply_complex(const MKL_Complex16 z1, const MKL_Complex16 z2) {
    MKL_Complex16 result;
    result.real = z1.real * z2.real - z1.imag * z2.imag;
    result.imag = z1.real * z2.imag + z1.imag * z2.real;
    return result;
}

void TransposeMatMKLComplex(const MKL_INT N, MKL_Complex16* Mat) {
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

void PrintArrDouble(const int N, const int M, const double* Arr) {
    for(int i=0; i<N; i++) {
        printf("%f, ", Arr[i]);
        if(i==M-1) printf("\n");
    }
    printf("\n");
}

void PrintArrMKLComplex(const int N, const int M, const MKL_Complex16* Arr) {
    for(int i=0; i<N; i++) {
        printf("%f+%f, ", Arr[i].real, Arr[i].imag);
        if(i==M-1) printf("\n");
    }
    printf("\n");
}