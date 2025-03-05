# This file contains most of the functions used in Nanodisks.py, Tests.py and 
# ExtCoeff.py. Defined are potenials, charge density calculations, Frude model 
# and all matrices for the calculations (D2, D1, G2, G1, G0 -> M).
# All parameters should be in the same unit system (preferably atomic units).

import numpy as np
import scipy.integrate as sci
from tqdm import tqdm

def FillTheta(N): return np.arange(0.5*1/N, 1, 1/N)

def phi_ext(ThetaArr, a): return -a*ThetaArr/2

def TotalPhi(phiArr, radius_index, angle): return phiArr[radius_index]*np.cos(angle)

def TotalChargeDensity(rhoArr, radius_index, m, angle): return 2*rhoArr[radius_index]*np.cos(m*angle)

def Drude(EF, omega, gamma): return 1j/np.pi * EF/(omega + 1j*gamma)# + 1/4 * (1j/np.pi * np.log(np.abs((omega-2*EF)/(omega+2*EF))))

def FillD1(N, m):
    delta = 1/N
    DArr = np.zeros((N, N))
    for i in range(1, N-1):
        DArr[i][i-1] = -1
        DArr[i][i+1] = 1
    
    # Lower boundary:
    m_term = -(-1)**m # m = even/odd -> m_term = -1/1
    DArr[0][0] = m_term
    DArr[0][1] = 1
    # Upper boundary:
    DArr[N-1][N-2] = -1
    DArr[N-1][N-1] = 1
    
    return DArr/(2*delta)

def FillD2(N, m):
    delta = 1/N
    D2Arr = np.zeros((N, N))
    for i in range(1, N-1):
        D2Arr[i][i-1] = 1
        D2Arr[i][i] = -2
        D2Arr[i][i+1] = 1
    
    # Lower boundary:
    m_term = 1 - (-1)**m # m = even/odd -> m_term = 0/2
    D2Arr[0][0] = -1-m_term
    D2Arr[0][1] = 1
    # Upper boundary:
    D2Arr[N-1][N-2] = 1
    D2Arr[N-1][N-1] = -1
    
    return D2Arr/delta**2

def TotalD(N, m, ThetaArr):
    D2 = FillD2(N, m)
    D1 = FillD1(N, m)

    G1 = np.zeros((N, N))
    G0 = np.zeros((N, N))
    for i in range(N):
        G1[i][i] = 1/ThetaArr[i]
        G0[i][i] = -m**2/ThetaArr[i]**2

    return D2 + np.dot(G1, D1) + G0

# def TotalD(N, m, ThetaArr):
#     D2 = FillD2(N, m)

#     G0 = np.zeros((N, N))
#     for i in range(N): G0[i][i] = -m**2/ThetaArr[i]**2

#     return D2 + G0

I_m0_table = np.load("I_m0.npy")
I_m1_table = np.load("I_m1.npy")

def PrintTables(index):
    print(I_m0_table[index])
    print(I_m1_table[index])
    return

def g_m(x, cutoff, m):
    res = 0
    x_max = 0.999
    if m==0:
        if x < x_max:
            for l in range(cutoff): res += x**(2*l) * I_m0_table[l]
        else:                       res = 2 * np.pi - 2 * np.log(1 - x**2)
    else:
        if x < x_max:
            for l in range(cutoff): res += x**(2*l+1) * I_m1_table[l]
        else:                       res = np.pi * x - 2 * x * np.log(1 - x**2)
    return res

def IntegrandG2(x, m, Theta_j, Theta_i, cutoff):
    S, L = min(Theta_i, x), max(Theta_i, x)
    # return x/L * g_m(S/L, cutoff, m) * (1 + (x - Theta_j)/x - (m**2 * (x - Theta_j)**2)/(2*x**2))
    return 1/L * g_m(S/L, cutoff, m) * (x + (x - Theta_j) - (m**2 * (x - Theta_j)**2)/(2*x))

def IntegrandG1(x, m, Theta_j, Theta_i, cutoff):
    S, L = min(Theta_i, x), max(Theta_i, x)
    # return x/L * g_m(S/L, cutoff, m) * (1/x - (m**2 * (x - Theta_j))/(x**2))
    return 1/L * g_m(S/L, cutoff, m) * (1 - (m**2 * (x - Theta_j))/x)

def IntegrandG0(x, m, Theta_j, Theta_i, cutoff):
    S, L = min(Theta_i, x), max(Theta_i, x)
    return -m**2/(L*x) * g_m(S/L, cutoff, 1)

def FillG2(ThetaArr, m, N, cutoff):
    delta = 1/N
    G2Arr = np.zeros((N, N))
    for i in tqdm(range(N)):
        for j in range(N):
            G2Arr[i][j] = sci.quad(IntegrandG2, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, points=[ThetaArr[j]], args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=500, epsabs=1e-10, epsrel=1e-10)[0]
            # GArr[i][j] += sci.quad(IntegrandG, ThetaArr[j], ThetaArr[j]+delta/2, args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=100)[0]
    return G2Arr

def FillG1(ThetaArr, m, N, cutoff):
    delta = 1/N
    G1Arr = np.zeros((N, N))
    for i in tqdm(range(N)):
        for j in range(N):
            G1Arr[i][j] = sci.quad(IntegrandG1, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, points=[ThetaArr[j]], args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=500, epsabs=1e-10, epsrel=1e-10)[0]
            # GTildeArr[i][j] += sci.quad(IntegrandGTilde, ThetaArr[j], ThetaArr[j]+delta/2, args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=100)[0]
    return G1Arr

def FillG0(ThetaArr, m, N, cutoff):
    delta = 1/N
    G0Arr = np.zeros((N, N))
    if m==0: return G0Arr
    for i in tqdm(range(N)):
        for j in range(N):
            G0Arr[i][j] = sci.quad(IntegrandG0, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, points=[ThetaArr[j]], args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=500, epsabs=1e-10, epsrel=1e-10)[0]
            # GTildeTildeArr[i][j] += sci.quad(IntegrandGTildeTilde, ThetaArr[j], ThetaArr[j]+delta/2, args=(m, ThetaArr[j], ThetaArr[i], cutoff), limit=100)[0]
    return G0Arr

def FillM(N, m, ThetaArr, cutoff):
    print('Start filling D1')
    D1 = FillD1(N, m)
    print('Start filling D2')
    D2 = FillD2(N, m)
    print('Start filling G2')
    G2 = FillG2(ThetaArr, m, N, cutoff)
    print('Start filling G1')
    G1 = FillG1(ThetaArr, m, N, cutoff)
    print('Start filling G0')
    G0 = FillG0(ThetaArr, m, N, cutoff)
    return np.dot(G2, D2) + np.dot(G1, D1) + G0

# def FillM(N, m, ThetaArr, cutoff):
#     D2 = FillD2(N, m)
#     G2 = FillG2(ThetaArr, m, N, cutoff)
#     G0 = FillG0(ThetaArr, m, N, cutoff)
#     return np.dot(G2, D2) + G0



# def analy(l, m):
#     if(m==0 and l%2==0):
#         n = int(l/2)
#         res = I_m0_table[n]
#     elif(m==1 and l%2==1):
#         n = int((l-1)/2)
#         res = I_m1_table[n]
#     else: res = 0
#     return res



# def g_m(x, cutoff, m):
#     res = 0
#     x_max = 0.999
#     if m==0:
#         if x < x_max:
#             for l in range(cutoff):
#                 res += x**(2*l) * I_m0_table[l] #I_m0(l)
#         else:
#             res = 2 * np.pi - 2 * np.log(1 - x**2)
#     else:
#         if x < x_max:
#             for l in range(cutoff):
#                 res += x**(2*l+1) * I_m1_table[l] #I_m1(l)
#         else:
#             res = np.pi * x - 2 * x * np.log(1 - x**2)
#     return res



# --- These have been calculated using the functions at the bottom ---
# l0_new  = [6.28318531, 1.57079633, 0.88357293, 0.61359231, 0.46978162, 0.38052311, 0.31974511, 0.27569859, 0.24231322, 0.21613741, 0.19506401, 0.17773394, 0.16323135, 0.15091656, 0.14032929, 0.13112993, 0.12306237, 0.11592986, 0.10957876, 0.10388735, 0.09875791, 0.09411113, 0.08988197, 0.08601654, 0.08246984, 0.07920404, 0.07618702, 0.0733914 , 0.07079369, 0.06837357, 0.06611344, 0.06399795, 0.06201364, 0.06014867, 0.0583926 , 0.05673616, 0.0551711 , 0.05369006, 0.05228647, 0.05095438, 0.04968848, 0.04848396, 0.04733645, 0.04624201, 0.04519702, 0.04419822, 0.04324261, 0.04232745, 0.04145022, 0.04060862, 0.0398005 , 0.03902392, 0.03827708, 0.03755827, 0.03686597, 0.03619873, 0.0355552 , 0.03493417, 0.03433445, 0.03375497, 0.03319474, 0.03265279, 0.03212825, 0.03162031, 0.03112817, 0.03065112, 0.03018847, 0.02973957, 0.02930384, 0.02888067, 0.02846957, 0.02807, 0.02768149, 0.0273036 , 0.02693588, 0.02657792, 0.02622937, 0.02588983, 0.02555897, 0.02523647, 0.02492199, 0.02461527, 0.02431599, 0.02402391, 0.02373877, 0.02346031, 0.02318831, 0.02292254, 0.0226628 , 0.02240888, 0.02216058, 0.02191772, 0.02168014, 0.02144764, 0.02122009, 0.0209973, 0.02077915, 0.02056549, 0.02035616, 0.02015107, 0.01995006, 0.01975302, 0.01955985, 0.0193704 , 0.0191846 , 0.01900232, 0.01882347, 0.01864797, 0.0184757 , 0.01830659, 0.01814055, 0.01797748, 0.01781733, 0.01766, 0.01750543, 0.01735354, 0.01720426, 0.01705753, 0.01691328, 0.01677145, 0.01663198, 0.01649481, 0.01635989, 0.01622715, 0.01609654, 0.01596804, 0.01584155, 0.01571707, 0.01559451, 0.01547386, 0.01535506, 0.01523806, 0.01512284, 0.01500936, 0.01489755, 0.01478741, 0.01467887, 0.01457192, 0.01446653, 0.01436263, 0.01426023, 0.01415927, 0.01405973, 0.01396159, 0.01386479, 0.01376935, 0.01367519, 0.01358232, 0.01349071, 0.01340031, 0.01331113, 0.01322312, 0.01313627, 0.01305056, 0.01296594, 0.01288243, 0.01279998, 0.01271858, 0.01263822, 0.01255885, 0.01248049, 0.01240308, 0.01232664, 0.01225114, 0.01217654, 0.01210286, 0.01203006, 0.01195813, 0.01188706, 0.01181682, 0.01174742, 0.01167881, 0.01161102, 0.011544, 0.01147775, 0.01141226, 0.0113475 , 0.01128348, 0.01122019, 0.01115759, 0.01109569, 0.01103447, 0.01097392, 0.01091404, 0.0108548 , 0.01079622, 0.01073824, 0.0106809 , 0.01062416, 0.01056802, 0.01051248, 0.0104575, 0.01040311, 0.01034928, 0.010296  , 0.01024327, 0.01019107, 0.01013941, 0.01008827, 0.01003763]
# l1_new  = [3.14159265, 1.17809724, 0.73631078, 0.53689328, 0.42280345, 0.34881285, 0.29690618, 0.25846743, 0.22885137, 0.20533053, 0.18619746, 0.17032836, 0.15695321, 0.14552668, 0.13565166, 0.12703212, 0.11944288, 0.11270959, 0.10669511, 0.10129016, 0.09640653, 0.09197225, 0.08792801, 0.08422452, 0.08082045, 0.07768089, 0.07477615, 0.07208085, 0.06957311, 0.06723401, 0.0650471 , 0.06299799, 0.06107404, 0.05926414, 0.05755842, 0.05594816, 0.05442555, 0.05298361, 0.05161612, 0.05031745, 0.04908252, 0.04790677, 0.04678603, 0.04571652, 0.04469483, 0.04371781, 0.04278258, 0.04188654, 0.04102726, 0.04020253, 0.0394103 , 0.03864869, 0.03791597, 0.03721051, 0.03653082, 0.03587553, 0.03524332, 0.03463301, 0.03404348, 0.03347368, 0.03292265, 0.03238946, 0.03187327, 0.03137328, 0.03088872, 0.03041891, 0.02996318, 0.02952089, 0.02909148, 0.02867439, 0.02826908, 0.02787508, 0.0274919 , 0.02711911, 0.0267563 , 0.02640307, 0.02605904, 0.02572387, 0.02539721, 0.02507874, 0.02476816, 0.02446517, 0.02416952, 0.02388091, 0.02359913, 0.02332391, 0.02305504, 0.0227923 , 0.02253548, 0.02228438, 0.02203882, 0.0217986 , 0.02156358, 0.02133356, 0.0211084 , 0.02088794, 0.02067204, 0.02046056, 0.02025336, 0.02005031, 0.0198513 , 0.01965619, 0.01946489, 0.01927728, 0.01909324, 0.01891269, 0.01873551, 0.01856163, 0.01839095, 0.01822337, 0.01805883, 0.01789722, 0.01773849, 0.01758255, 0.01742931, 0.01727874, 0.01713074, 0.01698525, 0.01684222, 0.01670157, 0.01656326, 0.01642721, 0.01629338, 0.01616172, 0.01603216, 0.01590467, 0.01577919, 0.01565566, 0.01553407, 0.01541434, 0.01529646, 0.01518035, 0.01506599, 0.01495335, 0.01484237, 0.01473304, 0.0146253 , 0.01451912, 0.01441449, 0.01431133, 0.01420966, 0.01410942, 0.01401057, 0.01391311, 0.01381698, 0.01372219, 0.01362868, 0.01353643, 0.01344544, 0.01335564, 0.01326705, 0.01317963, 0.01309334, 0.01300819, 0.01292412, 0.01284114, 0.01275922, 0.01267833, 0.01259848, 0.0125196 , 0.01244172, 0.01236481, 0.01228883, 0.01221379, 0.01213964, 0.0120664 , 0.01199405, 0.01192254, 0.01185189, 0.01178207, 0.01171307, 0.01164487, 0.01157746, 0.01151083, 0.01144495, 0.01137983, 0.01131545, 0.01125179, 0.01118885, 0.01112659, 0.01106504, 0.01100416, 0.01094394, 0.01088439, 0.01082547, 0.01076719, 0.01070954, 0.01065249, 0.01059606, 0.01054021, 0.01048496, 0.01043028, 0.01037616, 0.01032261, 0.0102696 , 0.01021714, 0.01016521, 0.0101138, 0.01006292, 0.01001254]
    
# def Total_mod0(Theta, l_max):
#     res = 0
#     for l in range(l_max): res += Theta**(2*l) * l0_new[l]
#     return res

# def Total_mod1(Theta, l_max):
#     res = 0
#     for l in range(l_max): res += Theta**(2*l+1) * l1_new[l]
#     return res

# # --- For the function g_m(x) I use the limit given in "Plasmons in inhomogeneously doped neutral and charged graphene nanodisks" for x close to 1 ---
# def g_0(x, cutoff):
#     if np.abs(x-1) >= 3e-3: return Total_mod0(x, cutoff)
#     else: return 2*np.pi - 2*np.log(1-x**2)

# def g_1(x, cutoff):
#     if np.abs(x-1) >= 3e-3: return Total_mod1(x, cutoff)
#     else: return x*np.pi - 2*x*np.log(1-x**2)

# def g_m(x, cutoff, m):
#     if m == 0: return g_0(x, cutoff)
#     else: return g_1(x, cutoff)

    
# def I_m0(n): return (2*np.pi)/(mp.power(16, n)) * (mp.factorial(2*n))**2/(mp.factorial(n))**4
# def I_m1(n): return np.pi/(mp.power(2, 4*n+1)) * (mp.factorial(2*n) * mp.factorial(2*n+2))/(mp.factorial(n)**2 * (mp.factorial(n+1))**2)


# import scipy.special as scs
# from mpmath import mp

# mp.dps = 50

# import math

# def g_m(mu, Lmax, m):
#     out = 0
#     mu_max = 0.999
    
#     if m == 0:
#         if mu < mu_max:
#             for l in range(Lmax + 1):
#                 I_lm = analy(l, 0)
#                 out += mu**l * I_lm
#         else:
#             out = 2 * np.pi - 2 * np.log(1 - mu**2)
    
#     elif m == 1:
#         if mu < mu_max:
#             for l in range(Lmax + 1):
#                 I_lm = analy(l, 1)
#                 out += mu**l * I_lm
#         else:
#             out = np.pi * mu - 2 * mu * np.log(1 - mu**2)
    
#     else:
#         out = 0
    
#     return out

# def analy(l, m):
#     if l % 2 == 0 and m == 0:
#         n = l // 2
#         out = (2 * np.pi / 16**n * math.factorial(2 * n)**2 / math.factorial(n)**4)
    
#     elif l % 2 == 1 and m == 1:
#         n = (l - 1) // 2
#         num = np.pi * math.factorial(2 * n) * math.factorial(2 * n + 2)
#         den = 2**(4 * n + 1) * math.factorial(n)**2 * math.factorial(n + 1)**2
#         out = num / den
    
#     else:
#         out = 0
    
#     return out


# Cannot use the code below, Bessel functions converge waaaay to slow...

# def IntBessel(q, Theta_i, Theta_p, l):
#     return scs.jv(l, q*Theta_i)*scs.jv(l, q*Theta_p)

# def IntV2(Theta_p, Theta_i, Theta_j, l):
#     BesselTerm = sci.quad(IntBessel, 0, np.inf, args=(Theta_i, Theta_p, l), limit=100)[0]
#     return BesselTerm * (Theta_p + (Theta_p-Theta_j) - l**2/(2*Theta_p)*(Theta_p-Theta_j)**2)

# def IntV1(Theta_p, Theta_i, Theta_j, l):
#     BesselTerm = sci.quad(IntBessel, 0, np.inf, args=(Theta_i, Theta_p, l), limit=100)[0]
#     return BesselTerm * (1 - l**2/Theta_p * (Theta_p-Theta_j))

# def IntV0(Theta_p, Theta_i, Theta_j, l):
#     BesselTerm = sci.quad(IntBessel, 0, np.inf, args=(Theta_i, Theta_p, l), limit=100)[0]
#     return - BesselTerm * l**2/Theta_p

# def FillV2(ThetaArr, l, N):
#     delta = 1/N
#     V2Arr = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             V2Arr[i][j] = sci.quad(IntV2, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, args=(ThetaArr[i], ThetaArr[j], l), limit=100)[0]
#     return V2Arr

# def FillV1(ThetaArr, l, N):
#     delta = 1/N
#     V1Arr = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             V1Arr[i][j] = sci.quad(IntV1, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, args=(ThetaArr[i], ThetaArr[j], l), limit=100)[0]
#     return V1Arr

# def FillV0(ThetaArr, l, N):
#     delta = 1/N
#     V0Arr = np.zeros((N, N))
#     if l==0: return V0Arr
#     for i in range(N):
#         for j in range(N):
#             V0Arr[i][j] = sci.quad(IntV0, ThetaArr[j]-delta/2, ThetaArr[j]+delta/2, args=(ThetaArr[i], ThetaArr[j], l), limit=100)[0]
#     return V0Arr

# def FillM_new(N, l, ThetaArr):
#     D = FillD1(N, l)
#     D2 = FillD2(N, l)
#     V2 = FillV2(ThetaArr, l, N)
#     V1 = FillV1(ThetaArr, l, N)
#     V0 = FillV0(ThetaArr, l, N)
#     return np.dot(V2, D2) + np.dot(V1, D) + V0



# def integrand0(x, l): return scs.legendre(l)(np.cos(x))
# def integrand1(x, l): return np.cos(x)*scs.legendre(l)(np.cos(x))

# def Integr0(l_max):
#     for l in range(l_max):
#         Integral = sci.quad(integrand0, -np.pi, np.pi, args=(l), limit=100)[0]
#         lookup0.append(Integral)
#     return

# def Integr1(l_max):
#     for l in range(l_max):
#         Integral = sci.quad(integrand1, -np.pi, np.pi, args=(l), limit=100)[0]
#         lookup1.append(Integral)
#     return

# def Integr0_diffInt(l_max):
#     for l in range(0, l_max, 2):
#         Integral = sci.quadrature(integrand0, -np.pi, np.pi, args=(l), maxiter=1000)[0]
#         lookup0.append(Integral)
#         print(Integral)
#     return

# def Integr1_diffInt(l_max):
#     for l in range(1, l_max, 2):
#         Integral = sci.quadrature(integrand1, -np.pi, np.pi, args=(l), maxiter=1000)[0]
#         lookup1.append(Integral)
#         print(Integral)
#     return

# Integr0(90)
# Integr1(90)
# print(lookup0)
# print(lookup1)

# def Total(Theta, l_max):
#     res = 0
#     for l in range(1, l_max, 2): #modified to work only for m=1
#         Integral = sci.quad(integrand, -np.pi, np.pi, args=(l), limit=100)[0]
#         res += Theta**l * Integral
#     return res

# lookup0 = []
# lookup1 = []





# # %%
# import scipy.special as scs
# # %%
# lookup0 = []
# lookup1 = []
# # %%
# def integrand0(x, l): return scs.legendre(l)(np.cos(x))
# def integrand1(x, l): return np.cos(x)*scs.legendre(l)(np.cos(x))

# def Integr0(l_max):
#     for l in range(0, l_max, 2):
#         Integral = sci.quad(integrand0, -np.pi, np.pi, args=(l), limit=100)[0]
#         lookup0.append(Integral)
#         print(Integral)
#     return

# def Integr0_diffInt(l_max):
#     for l in range(0, l_max, 2):
#         Integral = sci.quadrature(integrand0, -np.pi, np.pi, args=(l), maxiter=1000)[0]
#         lookup0.append(Integral)
#         print(Integral)
#     return

# def Integr1_diffInt(l_max):
#     for l in range(1, l_max, 2):
#         Integral = sci.quadrature(integrand1, -np.pi, np.pi, args=(l), maxiter=1000)[0]
#         lookup1.append(Integral)
#         print(Integral)
#     return

# def Integr1(l_max):
#     for l in range(1, l_max, 2):
#         Integral = sci.quad(integrand1, -np.pi, np.pi, args=(l), limit=100)[0]
#         lookup1.append(Integral)
#         print(Integral)
#     return

# order = 400
# # print('0:')
# # Integr0(order)
# # print('\n1:')
# # Integr1(order)
# # Integr0_diffInt(order)
# # Integr1_diffInt(order)
# # %%
# order = 400
# x0 = []
# x1 = []
# for i in range(0, order, 2): x0.append(i)
# for i in range(1, order, 2): x1.append(i)
# x0 = np.array(x0)
# x1 = np.array(x1)
# # plt.plot(x0, lookup0, 'o', markersize=3, label='0')
# # plt.plot(x0[1:], 4.9*(1.0*x1[1:])**-1.04)
# plt.plot(x1, lookup1, 'o', markersize=3, label='1')
# plt.plot(x1, 3.1*(1.0*x1)**-0.95)
# plt.legend()
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
# # %%
# print(np.array(lookup1))
# # %%
# [6.28318531, 1.57079633, 0.88357293, 0.61359231, 0.46978162, 0.38052311, 0.31974511, 0.27569859, 0.24231322, 0.21613741, 0.19506401, 0.17773394, 0.16323135, 0.15091656, 0.14032929, 0.13112993, 0.12306237, 0.11592986, 0.10957876, 0.10388735, 0.09875791, 0.09411113, 0.08988197, 0.08601654, 0.08246984, 0.07920404, 0.07618702, 0.0733914 , 0.07079369, 0.06837357, 0.06611344, 0.06399795, 0.06201364, 0.06014867, 0.0583926 , 0.05673616, 0.0551711 , 0.05369006, 0.05228647, 0.05095438, 0.04968848, 0.04848396, 0.04733645, 0.04624201, 0.04519702, 0.04419822, 0.04324261, 0.04232745, 0.04145022, 0.04060862, 0.0398005 , 0.03902392, 0.03827708, 0.03755827, 0.03686597, 0.03619873, 0.0355552 , 0.03493417, 0.03433445, 0.03375497, 0.03319474, 0.03265279, 0.03212825, 0.03162031, 0.03112817, 0.03065112, 0.03018847, 0.02973957, 0.02930384, 0.02888067, 0.02846957, 0.02807, 0.02768149, 0.0273036 , 0.02693588, 0.02657792, 0.02622937, 0.02588983, 0.02555897, 0.02523647, 0.02492199, 0.02461527, 0.02431599, 0.02402391, 0.02373877, 0.02346031, 0.02318831, 0.02292254, 0.0226628 , 0.02240888, 0.02216058, 0.02191772, 0.02168014, 0.02144764, 0.02122009, 0.0209973, 0.02077915, 0.02056549, 0.02035616, 0.02015107, 0.01995006, 0.01975302, 0.01955985, 0.0193704 , 0.0191846 , 0.01900232, 0.01882347, 0.01864797, 0.0184757 , 0.01830659, 0.01814055, 0.01797748, 0.01781733, 0.01766, 0.01750543, 0.01735354, 0.01720426, 0.01705753, 0.01691328, 0.01677145, 0.01663198, 0.01649481, 0.01635989, 0.01622715, 0.01609654, 0.01596804, 0.01584155, 0.01571707, 0.01559451, 0.01547386, 0.01535506, 0.01523806, 0.01512284, 0.01500936, 0.01489755, 0.01478741, 0.01467887, 0.01457192, 0.01446653, 0.01436263, 0.01426023, 0.01415927, 0.01405973, 0.01396159, 0.01386479, 0.01376935, 0.01367519, 0.01358232, 0.01349071, 0.01340031, 0.01331113, 0.01322312, 0.01313627, 0.01305056, 0.01296594, 0.01288243, 0.01279998, 0.01271858, 0.01263822, 0.01255885, 0.01248049, 0.01240308, 0.01232664, 0.01225114, 0.01217654, 0.01210286, 0.01203006, 0.01195813, 0.01188706, 0.01181682, 0.01174742, 0.01167881, 0.01161102, 0.011544, 0.01147775, 0.01141226, 0.0113475 , 0.01128348, 0.01122019, 0.01115759, 0.01109569, 0.01103447, 0.01097392, 0.01091404, 0.0108548 , 0.01079622, 0.01073824, 0.0106809 , 0.01062416, 0.01056802, 0.01051248, 0.0104575, 0.01040311, 0.01034928, 0.010296  , 0.01024327, 0.01019107, 0.01013941, 0.01008827, 0.01003763]
# # %%
# [3.14159265, 1.17809724, 0.73631078, 0.53689328, 0.42280345, 0.34881285, 0.29690618, 0.25846743, 0.22885137, 0.20533053, 0.18619746, 0.17032836, 0.15695321, 0.14552668, 0.13565166, 0.12703212, 0.11944288, 0.11270959, 0.10669511, 0.10129016, 0.09640653, 0.09197225, 0.08792801, 0.08422452, 0.08082045, 0.07768089, 0.07477615, 0.07208085, 0.06957311, 0.06723401, 0.0650471 , 0.06299799, 0.06107404, 0.05926414, 0.05755842, 0.05594816, 0.05442555, 0.05298361, 0.05161612, 0.05031745, 0.04908252, 0.04790677, 0.04678603, 0.04571652, 0.04469483, 0.04371781, 0.04278258, 0.04188654, 0.04102726, 0.04020253, 0.0394103 , 0.03864869, 0.03791597, 0.03721051, 0.03653082, 0.03587553, 0.03524332, 0.03463301, 0.03404348, 0.03347368, 0.03292265, 0.03238946, 0.03187327, 0.03137328, 0.03088872, 0.03041891, 0.02996318, 0.02952089, 0.02909148, 0.02867439, 0.02826908, 0.02787508, 0.0274919 , 0.02711911, 0.0267563 , 0.02640307, 0.02605904, 0.02572387, 0.02539721, 0.02507874, 0.02476816, 0.02446517, 0.02416952, 0.02388091, 0.02359913, 0.02332391, 0.02305504, 0.0227923 , 0.02253548, 0.02228438, 0.02203882, 0.0217986 , 0.02156358, 0.02133356, 0.0211084 , 0.02088794, 0.02067204, 0.02046056, 0.02025336, 0.02005031, 0.0198513 , 0.01965619, 0.01946489, 0.01927728, 0.01909324, 0.01891269, 0.01873551, 0.01856163, 0.01839095, 0.01822337, 0.01805883, 0.01789722, 0.01773849, 0.01758255, 0.01742931, 0.01727874, 0.01713074, 0.01698525, 0.01684222, 0.01670157, 0.01656326, 0.01642721, 0.01629338, 0.01616172, 0.01603216, 0.01590467, 0.01577919, 0.01565566, 0.01553407, 0.01541434, 0.01529646, 0.01518035, 0.01506599, 0.01495335, 0.01484237, 0.01473304, 0.0146253 , 0.01451912, 0.01441449, 0.01431133, 0.01420966, 0.01410942, 0.01401057, 0.01391311, 0.01381698, 0.01372219, 0.01362868, 0.01353643, 0.01344544, 0.01335564, 0.01326705, 0.01317963, 0.01309334, 0.01300819, 0.01292412, 0.01284114, 0.01275922, 0.01267833, 0.01259848, 0.0125196 , 0.01244172, 0.01236481, 0.01228883, 0.01221379, 0.01213964, 0.0120664 , 0.01199405, 0.01192254, 0.01185189, 0.01178207, 0.01171307, 0.01164487, 0.01157746, 0.01151083, 0.01144495, 0.01137983, 0.01131545, 0.01125179, 0.01118885, 0.01112659, 0.01106504, 0.01100416, 0.01094394, 0.01088439, 0.01082547, 0.01076719, 0.01070954, 0.01065249, 0.01059606, 0.01054021, 0.01048496, 0.01043028, 0.01037616, 0.01032261, 0.0102696 , 0.01021714, 0.01016521, 0.0101138, 0.01006292, 0.01001254]



# # N = 200
# # x = Fun.FillTheta(N)
# cutoff = np.arange(0, 5001, 1)
# mp.dps = 100

# def g(x, m, cutoff):
#     res = 0
#     if m==0:
#         for n in range(0, cutoff): res += x**(2*n) * I_m0(n)
#     else:
#         for n in range(0, cutoff): res += x**(2*n+1) * I_m1(n)
#     return res

# def I_m0(n): return (2*np.pi)/(mp.power(16, n)) * (mp.factorial(2*n))**2/(mp.factorial(n))**4

# def I_m1(n): return np.pi/(mp.power(2, 4*n+1)) * (mp.factorial(2*n) * mp.factorial(2*n+2))/(mp.factorial(n)**2 * (mp.factorial(n+1))**2)

# Arrrr = []
# for co in cutoff: Arrrr.append(np.float64(I_m1(co)))
# Arrrr = np.array(Arrrr)

# # Save the array to a .npy file
# # np.save("I_m1.npy", Arrrr)

# # Load the array back from the file
# # loaded_values = np.load("I_m1.npy")