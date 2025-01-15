#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#include "mkl.h"

#include "Constants.h"
#include "Functions.h"
#include "GeneralFunctions.h"

void writeDoubleToFile(const char* filename, int VecOrMat, int N, const double* Arr);

int main(const int argc, char** argv) {
    FILE *fp;
    char buffer[256];
    double d1 = -1, d2 = -1;

    // Run the A executable and open a pipe to read its output
    fp = popen("./Tests", "r");
    if (fp == NULL) {
        perror("Failed to run Tests");
        return 1;
    }

    // Read the output of A
    if (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Tokenize the string and extract the floats
        char *token;
        int count = 0;

        // Use strtok to split the string by ", "
        token = strtok(buffer, " ");
        while (token != NULL) {
            count++;
            if (count == 3) {
                d1 = atof(token); // Convert the third float
            } else if (count == 4) {
                d2 = atof(token); // Convert the fourth float
                break; // We have what we need, exit the loop
            }
            token = strtok(NULL, " ");
        }
    }

    // Close the pipe
    pclose(fp);

    printf("%f, %f\n", d1, d2);
    return 0;



    if(argc==1) {
        printf("This program calculates the extinction coefficients of a nanodisc for a given set of parameters.\n");
        printf("\nParameters:\nN:\t\tSize of Arrays and Matrices (NxN)\nm:\t\tAzimuthal Symmetrie\n"
               "cutoff:\t\tCutoff for g_m\nEF:\t\tFermi level in eV\nomega_S:\tStart of array of optical frequencies in eV\n"
               "omega_E:\tEnd of array of optical frequencies in eV\ngamma:\t\tDamping rate in ev\nradius:\t\tDiskradius in nm\n"
               "steps:\t\tNumber of steps through the array of frequencies\n\n"
               "As last argument the path to a folder can be given, where the output is saved.\n\n");
        printf("Extinction coefficients are saved in the file:\n"
               "(path to folder)ExtCoeff_N(Val)m(Val)cutoff(Val)EF(Val)omega_S(Val)omega_E(Val)gamma(Val)radius(Val)steps(Val).txt\n");
        return 1;
    }
    if(argc != 10 && argc != 11) {
        printf("Recheck arguments!\n");
        return 1;
    }

    // Read in parameters from command line
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

    // Prepare file for saving (Replace '.' with '_')
    char filename[512];
    char params[256];
    snprintf(params, sizeof(params), "N%dm%dcutoff%dEF%.2fomega_S%.2fomega_E%.2fgamma%.2fradius%.2fsteps%d",
             N, m, cutoff, EF*au_eV, omega_S*au_eV, omega_E*au_eV, gamma*au_eV, radius*au_nm, steps);
    for (char* p = params; *p; ++p) {
        if (*p == '.') { *p = '_'; }
    }
    if(argc==11) { snprintf(filename, sizeof(filename), "%sExtCoeff_%s.txt", argv[10], params); } //   ../DataEC/
    else{ snprintf(filename, sizeof(filename), "ExtCoeff_%s.txt", params); }
    printf("Output saved in:\n%s\n", filename);
    FILE* file = fopen(filename, "w");
    if(file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Allocate memory for all arrays (vectors and matrices)
    double* Theta_Vec               = (double*)mkl_malloc(sizeof(double)*N, 64);
    double* phi_ext_Vec             = (double*)mkl_malloc(sizeof(double)*N, 64);
    MKL_Complex16* phi_ext_Vec_c    = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N, 64);
    MKL_Complex16* CD_Vec           = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N, 64);
    MKL_INT* ipiv                   = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*N, 64);

    double* M_Mat                   = (double*)mkl_malloc(sizeof(double)*N*N, 64);
    double* D_Mat                   = (double*)mkl_malloc(sizeof(double)*N*N, 64);
    MKL_Complex16* D_Mat_c          = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N*N, 64);
    MKL_Complex16* Mtilde_Mat       = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16)*N*N, 64);

    // Fill vectors and matrices
    FillTheta(N, Theta_Vec);
    phi_ext(N, Theta_Vec, radius, phi_ext_Vec);
    FillM(N, m, Theta_Vec, cutoff, M_Mat);
    Dtilde(N, m, Theta_Vec, D_Mat);
    for(int i=0; i<N*N; i++) { D_Mat_c[i] = (MKL_Complex16){D_Mat[i], 0}; }

    // Loop over all values of omega in the range of omega_S to omega_E with (steps+1) many steps
    for(int i=0; i<steps+1; i++) {
        for(int j=0; j<N; j++) { phi_ext_Vec_c[j] = (MKL_Complex16){phi_ext_Vec[j], 0}; }

        const double omega = omega_S + (omega_E-omega_S)*(double)i/(double)steps;

        const MKL_Complex16 eta = (MKL_Complex16){creal(I*Drude(EF, omega, gamma)/(omega*radius)), cimag(I*Drude(EF, omega, gamma)/(omega*radius))};

        for(int j=0; j<N; j++) {
            for(int k=0; k<N; k++) {
                double eye = 0.0;
                if(j==k) eye = 1.0;
                Mtilde_Mat[j*N+k] = (MKL_Complex16){eye - multiply_complex(eta, (MKL_Complex16){M_Mat[j*N+k], 0}).real, - multiply_complex(eta, (MKL_Complex16){M_Mat[j*N+k], 0}).imag};
            }
        }

        // Calculate total potential phi from external potential
        /*
            nrhs: number of columns of phi_ext_Vec_c
            lda: leading order of Mtilde_Mat / number of columns in Mtilde_Mat
            ldb: leading order of phi_ext_Vec_c / number of columns in phi_ext_Vec_c
            Notice: Number of columns is leading order, since I use ROW_MAJOR (?)
            Documentation: https://gensoft.pasteur.fr/docs/lapack/3.9.0/d6/d10/group__complex16_g_esolve_ga531713dfc62bc5df387b7bb486a9deeb.html
        */
        const MKL_INT nrhs = 1, lda = N, ldb = 1;
        const MKL_INT info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, nrhs, Mtilde_Mat, lda, ipiv, phi_ext_Vec_c, ldb);
        if (info > 0) {
            printf("The diagonal element of the triangular factor of A,\nU(%lld,%lld) is zero, so that A is singular;\nthe solution could not be computed.\n", info, info);
            return 1;
        }
        if(info < 0) {
            printf("Error: Parameter %lld had an illegal value.\n", -info);
            return 1;
        }

        // Calculate charge density from total potential (saved in phi_ext_Vec_c from previous step)
        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta = {0.0, 0.0};
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &alpha, D_Mat_c, N, phi_ext_Vec_c, 1, &beta, CD_Vec, 1);

        // Use the approximation \int \rho * 1/(\Theta^2) d\Theta \approx \Delta\Theta * \sum_i \rho_i * 1/(\Theta_i^2)
        MKL_Complex16 integral_sol = (MKL_Complex16){0, 0};
        for(int j=0; j<N; j++) {
            integral_sol.real += CD_Vec[j].real * Theta_Vec[j] * Theta_Vec[j];
            integral_sol.imag += CD_Vec[j].imag * Theta_Vec[j] * Theta_Vec[j];
        }
        integral_sol.real *= 1.0/(double)N; integral_sol.imag *= 1.0/(double)N;

        // Calculate polarisation from integral
        const MKL_Complex16 Polarisation = multiply_complex((MKL_Complex16){2*M_PI*radius*radius, 0}, multiply_complex(eta, integral_sol));

        // Calculate the extinction coefficient per area from the polarisation
        const double ExtCo_Area = 4.0*omega/(radius*radius*c) * Polarisation.imag;

        // Save the value of omega and the extinction coefficient
        fprintf(file, "%.10f, %.10f", omega*au_eV, ExtCo_Area);
        if(i < steps) fprintf(file, "\n");
    }

    fclose(file);
    mkl_free(Theta_Vec);
    mkl_free(phi_ext_Vec);
    mkl_free(phi_ext_Vec_c);
    mkl_free(CD_Vec);
    mkl_free(ipiv);
    mkl_free(M_Mat);
    mkl_free(D_Mat);
    mkl_free(D_Mat_c);
    mkl_free(Mtilde_Mat);
    return 0;
}

void writeDoubleToFile(const char* filename, int VecOrMat, int N, const double* Arr) {
    FILE* file = fopen(filename, "w");
    if(file == NULL) {
        perror("Error opening file");
        return;
    }

    if(VecOrMat == 0) {
        for(int i=0; i<N; i++) {
            fprintf(file, "%.10f", Arr[i]);
            if(i < N-1) fprintf(file, "\n");
        }
    }
    else {
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                fprintf(file, "%.10f", Arr[i*N+j]);
                if(j < N-1) fprintf(file, ", ");
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}