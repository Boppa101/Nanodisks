# %%
import numpy as np
import matplotlib.pyplot as plt
import Functions as Fun
# %%
N = 100
m = 0
cutoff = 100
EF = 1/27.2114       # eV -> a.u. of energy
omega = 0.2/27.2114     # eV -> a.u. of energy
gamma = 0.02/27.2114    # eV -> a.u. of energy
radius = 25/0.0529177   # nm -> a.u. of length

ThetaArr = Fun.FillTheta(N)
Dtilde = Fun.TotalD(N, m, ThetaArr)

folder = 'Data/'
EVal_file = folder+'EVal_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
EVec_file = folder+'EVec_N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'
CD_file = folder+'CD___N'+str(N)+'m'+str(m)+'cutoff'+str(cutoff)+'EF'+"{:.2f}".format(EF*27.2114).replace('.', '_')+'omega'+"{:.2f}".format(omega*27.2114).replace('.', '_')+'gamma'+"{:.2f}".format(gamma*27.2114).replace('.', '_')+'radius'+"{:.2f}".format(radius*0.0529177).replace('.', '_')+'.txt'

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

# AllEVecs = np.zeros((N, N), dtype=complex)
# with open(EVec_file, 'r') as file:
#     c = 0
#     for line in file:
#         test = line.strip()
#         test = test.split(",")
#         for i in range(N):
#             ct = test[i].split("+")
#             AllEVecs[c, i] = float(ct[0]) + 1j*float(ct[1])
#         c+=1

# plt.plot(ThetaArr, np.real(AllEVecs[3]))
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

plt.plot(ThetaArr, np.real(AllCD[50]))
plt.show()
# %%
