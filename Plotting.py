# %%
import numpy as np
import matplotlib.pyplot as plt
import Functions as Fun
# %%
N = 600
m = 0
cutoff = 100
EF = 1/27.2114          # eV -> a.u. of energy
omega = 0.2/27.2114     # eV -> a.u. of energy
gamma = 0.02/27.2114    # eV -> a.u. of energy
radius = 25/0.0529177   # nm -> a.u. of length

eta = 1j*Fun.Drude(EF, omega, gamma)/(omega*radius)

ThetaArr = Fun.FillTheta(N)
Dtilde = Fun.TotalD(N, m, ThetaArr)
# %%
folder = 'Data/'
EVal_file = folder+'EVal_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
EVec_file = folder+'EVec_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
CD_file = folder+'CD___N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'

which = 1

# AllEVals = np.zeros(N, dtype=complex)
# with open(EVal_file, 'r') as file:
#     c = 0
#     for line in file:
#         test = line.strip()
#         test = test.split("+")
#         AllEVals[c] = float(test[0]) + 1j*float(test[1])
#         c+=1

# plt.plot(np.linspace(0, N, N), np.real(AllEVals), 'o')
# plt.show()

AllEVecs = np.zeros((N, N), dtype=complex)
with open(EVec_file, 'r') as file:
    c = 0
    for line in file:
        test = line.strip()
        test = test.split(",")
        for i in range(N):
            ct = test[i].split("+")
            AllEVecs[c, i] = float(ct[0]) + 1j*float(ct[1])
        c+=1

# plt.plot(ThetaArr, np.real(AllEVecs[which]))
# plt.show()

AllCD = np.zeros((N, N), dtype=complex)
with open(CD_file, 'r') as file:
    c = 0
    for line in file:
        test = line.strip()
        test = test.split(",")
        for i in range(N):
            ct = test[i].split("+")
            AllCD[c, i] = float(ct[0]) + 1j*float(ct[1])
        c+=1

x = np.append(-np.flip(ThetaArr), ThetaArr)
Pres = np.real(eta/radius * np.dot(Dtilde, AllEVecs[which]))
Pres = np.append((-1)**m*np.flip(Pres), Pres)
CRes = np.append((-1)**m*np.flip(np.real(AllCD[which])), np.real(AllCD[which]))
plt.plot(x, Pres)
plt.plot(x, CRes)
plt.show()
print(np.sum(np.real(AllCD[which]))/max(abs(np.real(AllCD[which]))))
# %%
N_P = 200
ThetaArr_P = Fun.FillTheta(N_P)

M_m0_noEta = Fun.FillM(N=N_P, m=0, ThetaArr=ThetaArr_P, cutoff=cutoff)
M_m1_noEta = Fun.FillM(N=N_P, m=1, ThetaArr=ThetaArr_P, cutoff=cutoff)
M_m0 = eta*M_m0_noEta
M_m1 = eta*M_m1_noEta

eigenvalues_m0, eigenvectors_m0 = np.linalg.eig(M_m0)
sorted_indices_m0 = np.argsort(eigenvalues_m0)
sorted_eigenvalues_m0 = eigenvalues_m0[sorted_indices_m0]
sorted_eigenvectors_m0 = eigenvectors_m0[:, sorted_indices_m0]

eigenvalues_m1, eigenvectors_m1 = np.linalg.eig(M_m1)
sorted_indices_m1 = np.argsort(eigenvalues_m1)
sorted_eigenvalues_m1 = eigenvalues_m1[sorted_indices_m1]
sorted_eigenvectors_m1 = eigenvectors_m1[:, sorted_indices_m1]

x_P = np.append(-np.flip(ThetaArr_P), ThetaArr_P)
# %%
m0_which = 1
m1_which = 0
y_m0 = sorted_eigenvectors_m0[:,m0_which]
y_m1 = sorted_eigenvectors_m1[:,m1_which]

D_m0 = Fun.TotalD(N_P, 0, ThetaArr_P)
D_m1 = Fun.TotalD(N_P, 1, ThetaArr_P)

CD_half_m0 = np.real(eta/radius*np.dot(D_m0, y_m0))
CD_m0 = np.append(np.flip(CD_half_m0), CD_half_m0)
if CD_m0[0] > 0: CD_m0 *= -1

CD_half_m1 = np.real(eta/radius*np.dot(D_m1, y_m1))
CD_m1 = np.append(-np.flip(CD_half_m1), CD_half_m1)
if CD_m1[0] * CD_m0[0] < 0: CD_m1 *= -1

# FullEVec_m0 = np.append(np.flip(y_m0), y_m0)
FullEVec_m0 = np.append(np.flip(AllEVecs[which]), AllEVecs[which])
grad_m0 = np.real(eta/radius*np.gradient(np.gradient(FullEVec_m0, 1/N, edge_order=2), 1/N, edge_order=2))
plt.plot(x, 3.5*grad_m0, label='Paper')
# FullEVec_m1 = np.append(-np.flip(y_m1), y_m1)
# FullEVec_m1 = np.append(-np.flip(AllEVecs[which]), AllEVecs[which])
# grad_m1 = np.real(eta/radius*np.gradient(np.gradient(FullEVec_m1, 1/N, edge_order=2), 1/N, edge_order=2))
# plt.plot(x, grad_m1, label='Paper')

plt.hlines(0, -1, 1, color='black', linestyle='--')
plt.plot(x_P, CD_m0, label='Python m=0')
# plt.plot(x_P, CD_m1, label='Python m=1')
plt.plot(x, 1.7*CRes, label='C (scaled)')
plt.legend()
# plt.ylim(-0.002, 0.002)
plt.show()
# %% EXCTINCTION COEFFICIENTS
def read_arrays(filename):
    all_omega = []
    all_ExtCoeff = []
    with open(filename, 'r') as file:
        for line in file:
            entries = line.strip().split(", ")
            all_omega.append(float(entries[0]))
            all_ExtCoeff.append(float(entries[1]))
    return all_omega, all_ExtCoeff
# %%
omega, ExtCoeff = read_arrays('DataEC/ExtCoeff_N500m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps600.txt')
print(np.max(ExtCoeff))
print(omega[np.where(ExtCoeff == np.max(ExtCoeff))[0][0]])
# %%
plt.plot(omega, ExtCoeff, color='red')
plt.xlim(0.15, 0.4)
plt.ylim(0, 1.5)
# plt.savefig('resC.png', dpi=350, bbox_inches='tight', transparent=True)
plt.show()
# %%
AllN = [50, 100, 200, 300, 400, 500]
ExtCoeff_max = [1.3369093479, 1.3395282491, 1.3407709807, 1.3411293254, 1.3412658275, 1.3414016051]
center = [0.25255, 0.2521, 0.25195, 0.25195, 0.25195, 0.2519]

plt.plot(AllN, ExtCoeff_max, 'o')
plt.title('Peak-value of extinction coefficient')
plt.xlabel('N')
plt.show()
plt.plot(AllN, center, 'o')
plt.title('Position of the Peak-value')
plt.xlabel('N')
plt.show()
# %%
