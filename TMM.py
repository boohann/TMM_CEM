# This code calculates Transfer Matrix Method, which was described by Raymond 
# C. Rumpf at the course Computational EM. I also used code of
# Sathyanarayan Rao from https://www.mathworks.com/matlabcentral/fileexchange/47637-transmittance-and-reflectance-spectra-of-multilayered-dielectric-stack-using-transfer-matrix-method
# 
import scipy.linalg as la
import numpy as np 

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
    SAB11 = SA11 + np.linalg.multi_dot([SA12, np.linalg.inv(Id -np.dot(SB11, SA22)), SB11, SA21])
    SAB12 = np.linalg.multi_dot([SA12, np.linalg.inv(Id - np.dot(SB11, SA22)), SB12])
    SAB21 = np.linalg.multi_dot([SB21, np.linalg.inv(Id - np.dot(SA22, SB11)), SA21])
    SAB22 = SB22 + np.linalg.multi_dot([SB21, np.linalg.inv(Id - np.dot(SA22, SB11)), SA22, SB12])
    return SAB11, SAB12, SAB21, SAB22; 


I = np.identity(2)
# units 
degrees = np.pi/180

### Device parameters ###
n1 = 3.2
n2 = 3.1
u1 = 1
u2 = 2
L1 = 7.87
L2 = 1.1
repeats = 20                            # Number of repeats
N_l = 2                                 # Number of layers per repeat

### Calculate matrices of layers of inputs ###
H_perm = np.array([u1, u2])
E_perm = np.array([n1**2, n2**2])
UR = np.repeat(H_perm, repeats)         # array of permeabilities in each layer 
ER = np.repeat(E_perm, repeats)         # array of permittivities in each layer  
Lu  = np.array([L1, L2]) 
L = np.repeat(Lu, repeats)
Lm = [i*1e-6 for i in L]
N  = repeats*N_l                        # Total number of layers

# external
er1 = n1**2 # reflection side medium
ur1 = 1.0
er2 = 1.0  # transition side medium
ur2 = 1.0
ni  = np.sqrt(er1) # refrctive index of incidence media

# source
lam0_nm  = np.linspace(1530, 1575) #free space wavelength 
lam0_m = [i*1e-9 for i in lam0_nm]
k0    = [(2*np.pi)/i for i in lam0_m]
theta = 0 * degrees #elevation angle 
phi   = 0 * degrees #azimuthal angle 
pte   = 1/np.sqrt(2)
ptm   = complex(0, pte)

# TMM algorithm
# Transverse Wave Vectors
kx = 0#[i*ni for i in k0]
ky = 0#[i*ni for i in k0]
kz = np.sqrt((k0**2)*ni-(kx**2)-(ky**2))


# ni li??
# homogeneous Gap Medium Parameters
Wh = I
# Qh = [kx*ky 1-kx*kx; ky*ky-1 -kx*ky] # Q
Qh = np.array([[kx*ky, 1-(kx**2)], [(ky**2)-1,  -kx*ky]])
Omh = complex(0, kz)*Wh # Omega
Vh = np.dot(Qh, np.linalg.inv(Omh))

# initialaze Global Scattering Matrix
Sg11 = np.zeros(shape = (2, 2)) 
Sg12 = I
Sg21 = I
Sg22 = np.zeros(shape = (2, 2))

## reflection side
krz = np.sqrt(ur1*er1 - (kx**2) - (ky**2)) 
Qr = 1/ur1 * np.array([[kx*ky, ur1*er1-kx**2], [ky**2-ur1*er1, -kx*ky]]) 
Omr = ni*krz*I
Vr = np.dot(Qr, np.linalg.inv(Omr)) 
Ar = I + np.dot(np.linalg.inv(i), j) for i,j in zip(Vh, Vr)] #!!!!!!!!!!!! check out this place later
Br = [I - np.dot(np.linalg.inv(i), j) for i,j in zip(Vh, Vr)] 
Sr11 = [np.dot(np.linalg.inv(-1*i), j) for i,j in zip(Ar, Br)]
Sr12 = [2 * np.linalg.inv(i) for i in Ar]
Sr21 = [0.5 * (i - np.linalg.multi_dot([j, np.linalg.inv(i),  j])) for i,j in zip(Ar, Br)]
Sr22 = [np.dot(j, np.linalg.inv(i)) for i,j in zip(Ar, Br)]

print(Sr11[1])

# connect external reflection region to device
for i in range(len(lam0_m)):
    Sg11[i], Sg12[i], Sg21[i], Sg22[i] = redhefferstar(Sr11[i], Sr12[i], Sr21[i], Sr22[i], Sg11[i], Sg12[i], Sg21[i], Sg22[i])


for j in range(len(lam0_m)):
    # reflection + each layer for every wavelength
    for i in range(N):
        kz = np.sqrt(ER[i]*UR[i] - (kx[j]**2) - (ky[j]**2))
        Q = 1/UR[i] * np.array([[kx[j]*ky[j], UR[i]*ER[i]-kx[j]**2], [ky[j]**2-UR[i]*ER[i], -kx[j]*ky[j]]])
        Om = ni*kz[i]*I
        V = np.dot(Q[i], np.linalg.inv(Qm[j]))
        A = I + np.dot(np.linalg.inv(V[i]), Vh[j])
        B = I - np.dot(np.linalg.inv(V[i]), Vh[j])
        X = la.expm(Om[i]*k0[j]*Lm[i])
        D = A[i] - np.linalg.multi_dot([X[i], B[i], np.linalg.inv(A[i]),X[i], B[i]])
        S11 = np.dot(np.linalg.inv(D[i]), np.linalg.multi_dot([X[i], B[i], np.linalg.inv(A[i]), X[i], A[i]]) - B[i])
        S22 = S11
        S12 = np.linalg.multi_dot([np.linalg.inv(D[i]), X[i], A[i] - np.linalg.multi_dot([B[i], np.linalg.inv(A[i]), B[i]])])
        S21 = S12
        # update gloabal S-matrix
        
        Sg11, Sg12, Sg21, Sg22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, S11, S12, S21, S22)
    
    # Now we add transmission
    ktz = np.sqrt(ur2*er2 - (kx[j]**2)-(ky[j]**2))
    Qt = 1/ur2 * np.array([[kx[j]*ky[j], ur2*er2-(kx[j]**2)], [(ky[j]**2)-ur2*er2, -kx[j]*ky[j]]])
    Omt = ni*ktz[j]*I
    Vt = np.dot(Qt[j], np.linalg.inv(Omt[j]))
    At = I + np.dot(np.linalg.inv(Vh[j]), Vt[j])
    Bt = I - np.dot(np.linalg.inv(Vh[j]), Vt[j])
    St11 = np.dot(Bt[j], np.linalg.inv(At[j]))
    St12 = 0.5*(At[j] - np.linalg.multi_dot([Bt[j], np.linalg.inv(At[j]), Bt[j]]))
    St21 = 2*np.linalg.inv(At[j])
    St22 = -1*np.dot(np.linalg.inv(At[j]), Bt[j])
    
    Sf11, Sf12, Sf21, Sf22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, St11, St12, St21, St22)
    
    # Source
    Kinc = np.dot(k0[j]*ni, np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]))
    ncn = np.array([0, 0, -1]) # surface normal
    aTE = np.array([0, 1, 0])
    
    
    aTM = np.cross(aTE, Kinc)/np.linalg.norm(np.cross(aTE, Kinc))
    P = pte*aTE + ptm*aTM
    cinc = np.array([[P[0]], [P[1]]])
    
    # Tranmitted and reflected fields
    Er  = np.dot(Sf11, cinc)
    Et  = np.dot(Sf21, cinc)
    Erx = Er[0]
    Ery = Er[1]
    Etx = Et[0]
    Ety = Et[1]
    

    # Longitudial field components
    Erz = -(kx[j]*Erx[j] + ky[j]*Ery[j])/krz[j]
    Etz = -(kx[j]*Etx[j] + ky[j]*Ety[j])/ktz[j]
    
    # Transmittance and reflectance
    R = abs(Erx[j])**2 + abs(Ery[j])**2 + abs(Erz[j])**2
    T = (abs(Etx[j])**2 + abs(Ety[j])**2 + abs(Etz[j])**2)*np.real((ur1*ktz[j])/(ur2*krz[j]))
    
    Tx[j] = Tx.append(abs(T))
    Rx[j] = Rx.append(abs(R))

    print(T)
'''
% plot result
figure(1)
plot(lam0,Tx,lam0,Rx,'r','linewidth',2)
axis([400e-9 800e-9 -0.05 1.05])
xlabel('\lambda','fontsize',14)
ylabel('T (in blue) or R (in red)','fontsize',14)
title('Transmittance (T) and Reflectance(R)','fontsize',14)
fh = figure(1);
set(fh, 'color', 'white');

function [SAB11,SAB12,SAB21,SAB22] = star_product (SA11,SA12,SA21,SA22,SB11,SB12,SB21,SB22)
% calculates star procudtion for two scattering matrix
    I = eye(2);
    D = SA12 * (I - SB11*SA22)^-1;
    F = SB21 * (I - SA22*SB11)^-1;

    SAB11 = SA11 + D*SB11*SA21;
    SAB12 = D*SB12;
    SAB21 = F*SA21;
    SAB22 = SB22 + F*SA22*SB12;
end
'''
