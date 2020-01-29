# This code calculates Transfer Matrix Method, which was described by Raymond 
# C. Rumpf at the course Computational EM. I also used code of
# Sathyanarayan Rao from https://www.mathworks.com/matlabcentral/fileexchange/47637-transmittance-and-reflectance-spectra-of-multilayered-dielectric-stack-using-transfer-matrix-method
# 
import scipy.linalg as la
import numpy as np 
import matplotlib.pyplot as plt

Tx = []
Rx = []

def redhefferstar(SA11, SA12, SA21, SA22, SB11, SB12, SB21, SB22):
    # redhefferstar product combines two scattering matrices to form a overall
    # scattering matrix. It is used in forming scattering matrices of
    # dielectric stacks in transfer matrix method
    # SA and SB are  scattering matrices of two different layers
    # and this function outputs
    # SAB which is the combined scaterring matrix of two layers
    Num = len(SA11)
    Id = np.identity(Num) 
    SAB11 = SA11 + np.linalg.multi_dot([SA12, np.linalg.inv(Id - np.dot(SB11, SA22)), SB11, SA21])
    SAB12 = np.linalg.multi_dot([SA12, np.linalg.inv(Id - np.dot(SB11, SA22)), SB12])
    SAB21 = np.linalg.multi_dot([SB21, np.linalg.inv(Id - np.dot(SA22, SB11)), SA21])
    SAB22 = SB22 + np.linalg.multi_dot([SB21, np.linalg.inv(Id - np.dot(SA22, SB11)), SA22, SB12])
    return SAB11, SAB12, SAB21, SAB22 


# Units 
I = np.identity(2)
degrees = np.pi/180
lam0_nm  = np.linspace(1200, 1600, 1000)       # Free space wavelength 
lam0_m   = [i*1e-9 for i in lam0_nm]

### Device parameters ###
n1 = 3
n2 = 4
u1 = 1
u2 = 2
L1 = 7.87
L2 = 1.1
repeats = 3                             # Number of repeats
N_l = 2                                 # Number of layers per repeat
N  = repeats*N_l                        # Total number of layers

### Calculate matrices of layers of inputs ###
H_perm = np.array([u2, u1])
E_perm = np.array([n2**2, n1**2])
UR = np.tile(H_perm, N)                 # Array of permeabilities in each layer 
ER = np.tile(E_perm, N)                 # Array of permittivities in each layer  
Lu  = np.array([L2, L1]) 
L = np.tile(Lu, N)
Lm = [i*1e-6 for i in L]

### External ###
er1 = n1**2                             # Reflection side medium
ur1 = 1.0
er2 = 1.0                               # Transition side medium
ur2 = 1.0
ni  = np.sqrt(er1)                      # Refrctive index of incidence media

### Source ###
theta    = 0 * degrees                  # Elevation angle 
phi      = (np.pi/2) * degrees          # Azimuthal angle 
pte      = 1/np.sqrt(2)
ptm      = complex(0, pte)

### TE plane wave ransverse wave vectors ###
kx = 0
ky = 0

for j in range(len(lam0_m)):
   
    k0 =(2*np.pi)/lam0_m[j] 
    kz = np.sqrt((k0**2)*ni-(kx**2)-(ky**2)) 
    

    # Homogeneous gap medium parameters
    Wh = I
    # Qh = [kx*ky 1-kx*kx; ky*ky-1 -kx*ky] # Q
    Qh = np.array([[kx*ky, 1-(kx**2)], [(ky**2)-1,  -kx*ky]])
    Omh = complex(0, kz)*Wh  # Omega
    Vh = np.dot(Qh, np.linalg.inv(Omh))

    #####################
    ### TMM algorithm ###
    #####################


    ### Initialaze global scattering matrix ###
    Sg11 = np.zeros(shape = (2, 2)) 
    Sg12 = I
    Sg21 = I
    Sg22 = np.zeros(shape = (2, 2)) 


    ### Reflection side ###
    krz  = np.sqrt(ur1*er1 - (kx**2) - (ky**2)) 
    Qr   = 1/ur1 * np.array([[kx*ky, ur1*er1-kx**2], [ky**2-ur1*er1, -kx*ky]]) 
    Omr  = ni*krz*I
    Vr   = np.dot(Qr, np.linalg.inv(Omr)) 
    Ar   = I + np.dot(np.linalg.inv(Vh), Vr)  #!!!!!!!!!!!! check out this place later
    Br   = I - np.dot(np.linalg.inv(Vh), Vr)  
    Sr11 = np.dot(np.linalg.inv(-1*Ar), Br)
    Sr12 = 2 * np.linalg.inv(Ar) 
    Sr21 = 0.5 * (Ar - np.linalg.multi_dot([Br, np.linalg.inv(Ar),  Br]))
    Sr22 = np.dot(Br, np.linalg.inv(Ar)) 

    ### Connect external reflection region to device ###
    Sg11, Sg12, Sg21, Sg22 = redhefferstar(Sr11, Sr12, Sr21, Sr22, Sg11, Sg12, Sg21, Sg22)


    ### Reflection + each layer for every wavelength ###
    for i in range(N):
        kz = np.sqrt(ER[i]*UR[i] - (kx**2) - (ky**2))
        Q = 1/UR[i] * np.array([[kx*ky, UR[i]*ER[i]-kx**2], [ky**2-UR[i]*ER[i], -kx*ky]])
        Om = ni*kz*I
        V = np.dot(Q, np.linalg.inv(Om))
        A = I + np.dot(np.linalg.inv(V), Vh)
        B = I - np.dot(np.linalg.inv(V), Vh)
        X = la.expm(Om*k0*Lm[i])
        D = A - np.linalg.multi_dot([X, B, np.linalg.inv(A),X, B])
        S11 = np.dot(np.linalg.inv(D), np.linalg.multi_dot([X, B, np.linalg.inv(A), X, A]) - B)
        S22 = S11
        S12 = np.linalg.multi_dot([np.linalg.inv(D), X, A - np.linalg.multi_dot([B, np.linalg.inv(A), B])])
        S21 = S12
        ### Update gloabal S-matrix
        Sg11, Sg12, Sg21, Sg22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, S11, S12, S21, S22)
    
    ### Now we add transmission ###
    ktz = np.sqrt(ur2*er2 - (kx**2)-(ky**2))
    Qt = 1/ur2 * np.array([[kx*ky, ur2*er2-(kx**2)], [(ky**2)-ur2*er2, -kx*ky]])
    Omt = ni*ktz*I
    Vt = np.dot(Qt, np.linalg.inv(Omt))
    At = I + np.dot(np.linalg.inv(Vh), Vt)
    Bt = I - np.dot(np.linalg.inv(Vh), Vt)
    St11 = np.dot(Bt, np.linalg.inv(At))
    St12 = 0.5*(At - np.linalg.multi_dot([Bt, np.linalg.inv(At), Bt]))
    St21 = 2*np.linalg.inv(At)
    St22 = -1*np.dot(np.linalg.inv(At), Bt)
    
    Sf11, Sf12, Sf21, Sf22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, St11, St12, St21, St22)
    ### Source ###
    Kinc = np.dot(k0*ni, np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))
    ncn = np.array([0, 0, -1]) # Surface normal
    aTE = np.array([0, 1, 0])
    
    
    aTM = np.cross(aTE, Kinc)/np.linalg.norm(np.cross(aTE, Kinc))
    P = pte*aTE + ptm*aTM
    cinc = np.array([[P[0]], [P[1]]])
    

    ### Tranmitted and reflected fields ###
    Er  = np.dot(Sf11, cinc)
    Et  = np.dot(Sf21, cinc)
    Erx = Er[0]
    Ery = Er[1]
    Etx = Et[0]
    Ety = Et[1]
    

    ### Longitudial field components ###
    Erz = -(kx*Erx + ky*Ery)/krz
    Etz = -(kx*Etx + ky*Ety)/ktz
 

    ### Transmittance and reflectance ###
    R = abs(Erx)**2 + abs(Ery)**2 + abs(Erz)**2
    T = (abs(Etx)**2 + abs(Ety)**2 + abs(Etz)**2)*np.real((ur1*ktz)/(ur2*krz))
    Tx.append(T)
    Rx.append(R)

print(Rx)
### Plotting ###
plt.plot(lam0_nm, Rx)
plt.title('Bragg Grating')
plt.xlabel('WL (nm)')
plt.ylabel('Reflec')
plt.show()
