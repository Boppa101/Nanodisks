#include <math.h>

#include "Constants.h"

// ------ Physical Constants ------

const double hbar         = 6.626e-34/(2.0*M_PI);     // hbar in J*s
const double h_eV         = 4.13567e-15;              // h in eV*s
const double hbar_eV      = h_eV/(2.0*M_PI);          // hbar in eV*s
const double qe           = 1.602176634e-19;          // Electron charge in C
const double m_elec       = 9.10938356e-31;           // Electron mass in kg
const double cc           = 299792458;                // Speed of light in m/s
const double eps_0        = 8.8541878128e-12;         // Vacuum permittivity
const double c_au         = 137.03599971;             // Speed of light in a.u. of speed
const double a0_au        = 0.529177208;              // Bohr radius a0 in Angstroms
const double a0_nm        = 1e-1*a0_au;               // Bohr radius a0 in nm
const double a0_cm        = 1e-8*a0_au;               // Bohr radius a0 in cm
const double a0_m         = 1e-10*a0_au;              // Bohr radius a0 in m
const double kB_J         = 1.380649e-23;             // Boltzmann constant in J/K
const double kB_eV        = 8.617333262145e-5;        // Boltzmann constant in eV/K

const double graphene_vF  = c_au/300;                 // Graphene Fermi velocity in a.u. of speed
const double graphene_dg  = 3.3e-8;                   // Graphene thickness in cm


// ------ Conversion Factors ------

const double au_nm        = 0.05291772083;            // 1 a.u. of length in nm
const double au_kg        = 9.10938291e-31;           // 1 a.u. of mass in kg
const double au_s         = 2.418884326502e-17;       // 1 a.u. of time in s
const double fs           = 1e-15/au_s;               // 1 a.u. of a femto-second
const double au_eV        = 27.2113834;               // 1 a.u. of energy in eV
const double au_E         = 5.14221e11;               // 1 a.u. of electric field in V/m
const double au_J         = 4.3597442e-18;            // 1 a.u. of energy in J
const double au_T         = 2.35e5;                   // 1 a.u. of magnetic field in T
const double au_I         = qe/au_s;                  // 1 a.u. of intensity in Amperes
const double nm_au        = 1.0/au_nm;                // 1 nm in a.u. of length
const double nm           = nm_au;                    // Similar to Javier's notation?
const double s_au         = 1.0/au_s;                 // 1 second in a.u. of time
const double eV_au        = 1.0/au_eV;                // 1 eV in a.u. of energy
const double E_au         = 1.0/au_E;                 // 1 V/m in a.u. of electric field
// const double I_au         = pow(au_s, 3.0)/au_kg; // 1 W/m² in a.u. of intensity // This is problematic, since pow(...) is not constant
const double kb_au        = 3.1668151e-6;             // k_B in 1/(a.u. K)
const double T_au         = 1.0/au_T;                 // 1 T in a.u. of magnetic field