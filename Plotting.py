# %%
import numpy as np
import matplotlib.pyplot as plt
import Functions as Fun
# %% EIGENVECTORS
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
# folder = 'DataEC/'
# CD_file = folder+'ExtCoeff_N300m1cutoff101EF1_00omega_S0_25omega_E0_25gamma0_02radius50_00steps1T0_00op0.txt'
# CD_file = folder+'ExtCoeff_N300m1cutoff101EF1_00omega_S0_20omega_E0_20gamma0_02radius50_00steps1T0_00op0.txt'
folder = 'Data/'
# EVal_file = folder+'EVal_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
EVec_file = folder+'EVec_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
CD_file = folder+'CD___N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
# %%
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

# AllCD = np.zeros((N), dtype=complex)
# with open(CD_file, 'r') as file:
#     c = 0
#     for line in file:
#         test = line.strip()
#         ct = test.split("+")
#         AllCD[c] = float(ct[0]) + 1j*float(ct[1])
#         c+=1

# which = np.array([0])
which = np.array([0, 1, 2, 3])
x = np.append(-np.flip(ThetaArr), ThetaArr)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Standard colors

plt.hlines(which, -1, 1, color='grey', linestyle='--')
for i in which:
    renorm = np.max(np.abs(np.real(AllEVecs[i])))+1e-7
    CRes = np.append((-1)**m*np.flip(np.real(AllEVecs[i])), np.real(AllEVecs[i]))/(2.5*renorm)
    if (CRes[0] > 0 and i!=0): CRes *= -1
    plt.plot(x, CRes+i, linewidth=2, color=colors[i])

plt.xlabel(r'$R$', fontsize=14)
plt.ylabel(r'$\phi(R, \varphi)$ a.u.', fontsize=14)
plt.yticks(which, which+1)
plt.xticks([-1, -0.5, 0, 0.5, 1], [r'$-a$', r'$-a/2$', r'$0$', r'$a/2$', r'$a$'])
plt.xlim(-1.01, 1.01)
plt.ylim(-0.5, 3.8)
# plt.savefig('images/PotEVecm'+str(m)+'.png', dpi=350, bbox_inches='tight')
plt.show()

plt.hlines(which, -1, 1, color='grey', linestyle='--')
for i in which:
    # renorm = np.max(np.abs(np.real(AllCD)))+1e-7
    # CRes = np.append((-1)**m*np.flip(np.real(AllCD)), np.real(AllCD))
    renorm = np.max(np.abs(np.real(AllCD[i])))+1e-7
    CRes = np.append((-1)**m*np.flip(np.real(AllCD[i])), np.real(AllCD[i]))/renorm
    if (CRes[0] > 0): CRes *= -1
    plt.plot(x[2:-2], CRes[2:-2]+i, linewidth=2, color=colors[i])

plt.xlabel(r'$R$', fontsize=14)
plt.ylabel(r'$\rho(R, \varphi)$ a.u.', fontsize=14)
plt.yticks(which, which+1)
plt.xticks([-1, -0.5, 0, 0.5, 1], [r'$-a$', r'$-a/2$', r'$0$', r'$a/2$', r'$a$'])
plt.xlim(-1.01, 1.01)
plt.ylim(-0.5, 3.8)
# plt.savefig('images/CDEVecm'+str(m)+'.png', dpi=350, bbox_inches='tight')
plt.show()
# %%
# %% 3D Plotting
r_idx = np.arange(N)
radius = np.linspace(0, 1, N)
angles = np.linspace(0, 2*np.pi, 4*N)
r_idx, angle = np.meshgrid(r_idx[:-2], angles[:-2])

thisOne = 2-m

renorm = np.max(np.abs(np.real(AllCD[thisOne])))
z = Fun.TotalChargeDensity(np.real(AllCD[thisOne][:-2])/renorm, r_idx, m, angle)

x = radius[r_idx] * np.cos(angle)
y = radius[r_idx] * np.sin(angle)

fig = plt.figure(figsize=[10, 10])
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='seismic', antialiased=False, rstride=1, cstride=1, edgecolor='none')

ax.view_init(azim=280, elev=15)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel(r'$x/a$', fontsize=18)
ax.set_ylabel(r'$y/a$', fontsize=18)
ax.set_zlabel(r'$\rho$ a.u.', fontsize=18)

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
# ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
# ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
# ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)

# ax.axis('off')

# ax.contourf(x, y, z, cmap='seismic', antialiased=False, levels=200)
ax.contourf(x, y, z, zdir='z', offset=-1.2, cmap='seismic', antialiased=False, levels=200)

ax.set_zlim(-1.2, 1)

# ax.set_zlabel(r'$\rho(x, y)$')
# ax.set_zlim(-1, 1)
# plt.savefig('images/CD3DPlotm'+str(m)+'.png', dpi=350, bbox_inches='tight')
plt.close()
# %%
# %%
# N_P = 200
# ThetaArr_P = Fun.FillTheta(N_P)

# M_m0_noEta = Fun.FillM(N=N_P, m=0, ThetaArr=ThetaArr_P, cutoff=cutoff)
# M_m1_noEta = Fun.FillM(N=N_P, m=1, ThetaArr=ThetaArr_P, cutoff=cutoff)
# M_m0 = eta*M_m0_noEta
# M_m1 = eta*M_m1_noEta

# eigenvalues_m0, eigenvectors_m0 = np.linalg.eig(M_m0)
# sorted_indices_m0 = np.argsort(eigenvalues_m0)
# sorted_eigenvalues_m0 = eigenvalues_m0[sorted_indices_m0]
# sorted_eigenvectors_m0 = eigenvectors_m0[:, sorted_indices_m0]

# eigenvalues_m1, eigenvectors_m1 = np.linalg.eig(M_m1)
# sorted_indices_m1 = np.argsort(eigenvalues_m1)
# sorted_eigenvalues_m1 = eigenvalues_m1[sorted_indices_m1]
# sorted_eigenvectors_m1 = eigenvectors_m1[:, sorted_indices_m1]

# x_P = np.append(-np.flip(ThetaArr_P), ThetaArr_P)
# # %%
# m0_which = 1
# m1_which = 0
# y_m0 = sorted_eigenvectors_m0[:,m0_which]
# y_m1 = sorted_eigenvectors_m1[:,m1_which]

# D_m0 = Fun.TotalD(N_P, 0, ThetaArr_P)
# D_m1 = Fun.TotalD(N_P, 1, ThetaArr_P)

# CD_half_m0 = np.real(eta/radius*np.dot(D_m0, y_m0))
# CD_m0 = np.append(np.flip(CD_half_m0), CD_half_m0)
# if CD_m0[0] > 0: CD_m0 *= -1

# CD_half_m1 = np.real(eta/radius*np.dot(D_m1, y_m1))
# CD_m1 = np.append(-np.flip(CD_half_m1), CD_half_m1)
# if CD_m1[0] * CD_m0[0] < 0: CD_m1 *= -1

# # FullEVec_m0 = np.append(np.flip(y_m0), y_m0)
# FullEVec_m0 = np.append(np.flip(AllEVecs[which]), AllEVecs[which])
# grad_m0 = np.real(eta/radius*np.gradient(np.gradient(FullEVec_m0, 1/N, edge_order=2), 1/N, edge_order=2))
# plt.plot(x, 3.5*grad_m0, label='Paper')
# # FullEVec_m1 = np.append(-np.flip(y_m1), y_m1)
# # FullEVec_m1 = np.append(-np.flip(AllEVecs[which]), AllEVecs[which])
# # grad_m1 = np.real(eta/radius*np.gradient(np.gradient(FullEVec_m1, 1/N, edge_order=2), 1/N, edge_order=2))
# # plt.plot(x, grad_m1, label='Paper')

# plt.hlines(0, -1, 1, color='black', linestyle='--')
# plt.plot(x_P, CD_m0, label='Python m=0')
# # plt.plot(x_P, CD_m1, label='Python m=1')
# # plt.plot(x, 1.7*CRes, label='C (scaled)')
# plt.legend()
# # plt.ylim(-0.002, 0.002)
# plt.show()
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
omega_D, ExtCoeff_D = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op0.txt')
omega_RPA, ExtCoeff_RPA = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op1.txt')
omega_RPAq1, ExtCoeff_RPAq1 = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op2q0_10.txt')
omega_RPAq2, ExtCoeff_RPAq2 = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op2q0_20.txt')
# omega_RPAq5, ExtCoeff_RPAq5 = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op2q0_50.txt')
# omega_D, ExtCoeff_D = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op0.txt')
# omega_DT, ExtCoeff_DT = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T500_00op0.txt')
# # omega_DI, ExtCoeff_DI = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op1.txt')
# # omega_DIT, ExtCoeff_DIT = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T300_00op1.txt')
# omega_RPA, ExtCoeff_RPA = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T0_00op2.txt')
# # omega_RPAT, ExtCoeff_RPAT = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T300_00op2.txt')
# omega_RPAT_1, ExtCoeff_RPAT_1 = read_arrays('DataEC/ExtCoeff_N100m1cutoff100EF1_00omega_S0_22omega_E0_28gamma0_02radius50_00steps400T500_00op2.txt')
# print(np.max(ExtCoeff), np.max(ExtCoeff_o))
# print(omega[np.where(ExtCoeff == np.max(ExtCoeff))[0][0]], omega_o[np.where(ExtCoeff_o == np.max(ExtCoeff_o))[0][0]])
# %%
plt.plot(omega_D, ExtCoeff_D, label='Drude')
plt.plot(omega_RPA, ExtCoeff_RPA, label='Local RPA')
plt.plot(omega_RPAq1, ExtCoeff_RPAq1, label='Non-Local RPA (q=0.1)', linestyle='--')
plt.plot(omega_RPAq2, ExtCoeff_RPAq2, label='Non-Local RPA (q=0.2)', linestyle='--')
# plt.plot(omega_RPAq5, ExtCoeff_RPAq5, label='Non-Local RPA (q=0.5)', linestyle='--')
# plt.plot(omega_DT, ExtCoeff_DT, label='Drude', linestyle='--')
# plt.plot(omega_DI, ExtCoeff_DI, label='Drude + inter')
# plt.plot(omega_DIT, ExtCoeff_DIT, label='Drude + inter', linestyle='--')
# plt.plot(omega_RPA, ExtCoeff_RPA, label='RPA')
# plt.plot(omega_RPAT_1, ExtCoeff_RPAT_1, label='RPA', linestyle='--')
# plt.plot(omega, ExtCoeff, color='blue', label='Drude')
# plt.plot(omega_n, ExtCoeff_n, color='red', label='Local RPA')
plt.legend()
plt.xlabel(r'$\hbar\omega\,(\text{eV})$', fontsize=14)
plt.ylabel(r'$\sigma^\text{ext}/\text{Area}$', fontsize=14)
# plt.xlim(0.15, 0.4)
# plt.ylim(0, 1.5)
# plt.savefig('ExtCoeff_Comparison.png', dpi=350, bbox_inches='tight')
plt.show()
# %%
# I used steps=600, omega_S=0.245 and omega_E=0.26
AllN = [50, 100, 200, 300, 400, 500]
ExtCoeff_max = [1.3369155247, 1.3395665935, 1.3407776365, 1.3411438215, 1.3413136131, 1.341410371]
center = [0.252525, 0.25215, 0.251975, 0.251925, 0.2519, 0.251875]

plt.plot(AllN, ExtCoeff_max, 'o')
plt.title('Peak-value of extinction coefficient')
plt.xlabel('N')
plt.show()
plt.plot(AllN, center, 'o')
plt.title('Position of the Peak-value')
plt.xlabel('N')
plt.show()
# %%