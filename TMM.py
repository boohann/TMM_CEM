'''
This code calculates Transfer Matrix Method, which was described by Raymond 
C. Rumpf at the course Computational EM. I also used code of
Sathyanarayan Rao from https://www.mathworks.com/matlabcentral/fileexchange/47637-transmittance-and-reflectance-spectra-of-multilayered-dielectric-stack-using-transfer-matrix-method
Updated to Python and applied to repeating laser Bragg grating by Niall Boohan [niall.boohan@tyndall.ie]

NOTES:
-TE plane wave assumed no kx or ky components ###
'''

import scipy.linalg as la
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import norm

##############################################################################
# Dashboard
##############################################################################
Tx = []
Rx = []
Zx = []

### Device parameters ###
lambda_0 = 1.550310559006211    # Selected wavelength (um)
n_rid = 3.12                    # Refractive index ridge
n_del = 0.0075                  # Reflective index step
n_def = n_rid-n_del             # Refractive index etched defect
u_rid = 1                       # Magnetic perm layer 1
u_def = 1                       # Magnetic perm layer 2
WL_start = 1550                 # Start WL (nm)
WL_stop  = 1560                 # Stop WL (nm)
sw = lambda_0/(4*n_def)         # Specify size of slot width
cl = 400                        # Cavity length
r_l = 0.5                       # Left facet reflectivity
r_r = 0.5                       # Right facet reflectivity
u = 1.0                         # General magnetic permeability

### Calc to turn reflectivites to n ###
n_l = (n_rid*(1-r_l))/(1+r_l) 
n_r = (n_rid*(1-r_r))/(1+r_r)

### Load slot positions from external calculation file ###
L = np.load('Pslot.npy') 
### Add right and left interface position to each slot ###
Lr = [L[i]-L[i-1]-sw for i in range(1, len(L))]
Lr.insert(0 , L[0]-sw/2)
Ls = [sw for i in range(len(L))]
Lm = [None]*(len(Lr)+len(Ls))
Lm[1::2] = Ls
Lm[::2] = Lr

##############################################################################
# Preliminary calculations
##############################################################################

### Applying calculation matrices ###
I = np.identity(2)                                  # Creating identity matrix
lam0_nm  = np.linspace(WL_start, WL_stop, 1000)     # Free space wavelength 
lam0_m   = [i*1e-9 for i in lam0_nm]

''' Test QW DFB stack
uR = [ 1, 1]
eR = [ 1.69**2, 1.46**2]
L  = [ 0.1479, 0.1712]
ER = np.tile(eR, 15)                     # Array of permittivities in each layer  
UR = np.tile(uR, 15)
Lm = np.tile(L, 15)                     # Array of permeabilities in each layer
'''
U = [1, 1]
E = [n_rid**2, n_def**2]
ER = np.tile(E, len(L))                     # Array of permittivities in each layer  
UR = np.tile(U, len(L))

### Reflectivity arrays for left and right facet coatings ###
l_E = np.array([n_l**2])                     
l_H = np.array([1])                             
r_E = np.array([n_r**2])                    
r_H = np.array([1])                             

### Insert left & right facets & end section of laser ###
Lm.insert(0,lambda_0/(2*n_def))             
Lm.append(cl-L[-1]-sw/2)
Lm.append(lambda_0/(2*n_def))

### Append facets & laser end section onto dielectric stack ###
# Laser end section
ER = np.append(ER, n_rid**2)
UR = np.append(UR, u_rid**2)
# Right
ER = np.append(ER, r_E)
UR = np.append(UR, r_H)
# Left
ER = np.insert(ER, 0, l_E)
UR = np.insert(UR, 0, l_H)

### Generate input & output boudaries for laser [set to inf] ###
erl = 1.0               # Left laser input
url = 1.0
err = 1.0               # Right laser output
urr = 1.0
ni  = np.sqrt(erl)                  # Refrctive index of incidence media

print(Lm)
print(ER)
Lm = [i*1e-6 for i in Lm]           # Convert lengths to meters

### Source ###
pte      = 1/np.sqrt(2)
ptm      = pte*1j


##############################################################################
# Combining matrices function
##############################################################################

### Redheffer matrix combination routine ###
def redhefferstar(SA11, SA12, SA21, SA22, SB11, SB12, SB21, SB22):
    Num = len(SA11)
    Id = np.identity(Num)
    DA = np.dot(SA12, np.linalg.inv(Id-np.dot(SB11, SA22)))
    FA = np.dot(SB21, np.linalg.inv(Id-np.dot(SA22, SB11)))
    SAB11 = SA11 + np.linalg.multi_dot([DA, SB11, SA21])
    SAB12 = np.dot(DA, SB12)
    SAB21 = np.dot(FA, SA21)
    SAB22 = SB22 + np.linalg.multi_dot([FA, SA22, SB12])
    return SAB11, SAB12, SAB21, SAB22 


##############################################################################
# TMM calculation section
##############################################################################

### Run code for range of wavelengths ###
for i in range(len(lam0_m)):
   
    k0 =(2*np.pi)/lam0_m[i] 
    kz = np.sqrt(1)                             #Apply refractive index of incident material

    # Homogeneous gap medium parameters
    Qh = np.array([[0, 1], [-1,  0]])
    Omh = kz*I*1j                     # Omega
    Vh = np.dot(Qh, np.linalg.inv(Omh))

    ### Initialaze global scattering matrix ###
    Sg11 = np.zeros(shape = (2, 2)) 
    Sg12 = I
    Sg21 = I
    Sg22 = np.zeros(shape = (2, 2)) 


    ### Add left external media ###
    krz  = np.sqrt(erl*url) 
    Qr = np.array([[0, erl*url], [-erl*url, 0]])            # Choose gap media of air
    Omr  = 1j*krz*I
    Vr   = Qr.dot(np.linalg.inv(Omr)) 
    Ar   = I + np.dot(np.linalg.inv(Vh), Vr)  
    Br   = I - np.dot(np.linalg.inv(Vh), Vr)  
    Sr11 = -1*np.dot(np.linalg.inv(Ar), Br)
    Sr12 = 2 * np.linalg.inv(Ar) 
    Sr21 = 0.5 * (Ar - np.linalg.multi_dot([Br, np.linalg.inv(Ar),  Br]))
    Sr22 = np.dot(Br, np.linalg.inv(Ar)) 

    ### Connect external reflection region to device ###
    Sg11, Sg12, Sg21, Sg22 = redhefferstar(Sr11, Sr12, Sr21, Sr22, Sg11, Sg12, Sg21, Sg22)


    ### Reflection + phase change for each layer for every wavelength ###
    for j in range(len(ER)):
        kz  = np.sqrt(ER[j]*UR[j])
        Q = np.array([[0, ER[j]*UR[j]], [-1*ER[j]*UR[j], 0]])         # Chose gap medium of air
        Om  = 1j*kz*I # Add loss in here #ni*kz*I
        V   = Q.dot(np.linalg.inv(Om))
        A   = I + np.dot(np.linalg.inv(V), Vh)
        B   = I - np.dot(np.linalg.inv(V), Vh)
        X   = la.expm(Om*k0*Lm[j])
        D = A - np.linalg.multi_dot([X, B, np.linalg.inv(A), X, B])
        S11 = np.dot(np.linalg.inv(D), np.linalg.multi_dot([X, B, np.linalg.inv(A), X,A])-B)
        S22 = S11
        S12 = np.linalg.multi_dot([np.linalg.inv(D), X, A-np.linalg.multi_dot([B, np.linalg.inv(A), B])])
        S21 = S12
        ### Update gloabal S-matrix
        Sg11, Sg12, Sg21, Sg22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, S11, S12, S21, S22)
        

    ### Add right external media ###
    ktz  = np.sqrt(err*urr)
    Qt   = np.array([[0, err*urr], [-1*err*urr, 0]])  # Chose gap medium of air
    Omt  = 1j*ktz*I 
    Vt   = Qt.dot(np.linalg.inv(Omt))
    At   = I + np.dot(np.linalg.inv(Vh), Vt)
    Bt   = I - np.dot(np.linalg.inv(Vh), Vt)
    St11 = np.dot(Bt, np.linalg.inv(At))
    St12 = 0.5*(At - np.linalg.multi_dot([Bt, np.linalg.inv(At), Bt]))
    St21 = 2*np.linalg.inv(At)
    St22 = -1*np.dot(np.linalg.inv(At), Bt)
    
    Sf11, Sf12, Sf21, Sf22 = redhefferstar(Sg11, Sg12, Sg21, Sg22, St11, St12, St21, St22)
    

    ### Source ###
    Kinc = [j*k0*ni for j in np.array([0, 0,  1])]
    ncn  = np.array([0, 0, -1]) # Surface normal
    aTE  = np.array([0, 1,  0])
    aTM = np.cross(aTE, Kinc)/norm(np.cross(aTE, Kinc), 2)
    P    = pte*aTE + ptm*aTM
    cinc = np.array([[P[0]], [P[1]]])
    

    ### Tranmitted and reflected fields ###
    Er  = np.dot(Sf11, cinc)
    Et  = np.dot(Sf21, cinc)
    Erx = Er[0]
    Ery = Er[1]
    Etx = Et[0]
    Ety = Et[1]
    

    ### Transmittance and reflectance ###
    R = abs(Erx)**2 + abs(Ery)**2 
    T = (abs(Etx)**2 + abs(Ety)**2)*(np.real((url*ktz)/np.real(urr*krz)))

    Tx.append(abs(T))
    Rx.append(abs(R))
    Z = R+T
    Zx.append(Z)

##############################################################################
# Visualisation
##############################################################################


### Plotting ###
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(lam0_nm, Tx, 'b-')
ax2.plot(lam0_nm, Tx, 'b-')

ax1.set_xlabel('WL (nm)')
ax1.set_title('TMM Bragg Grating')
ax1.set_ylabel('Ref', color='g')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='y', colors='g')
ax2.set_ylabel('Trans', color='b')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', colors='b')
plt.show()
### Calculate matrices of layers of Bragg grating of inputs ###

#Q   = (1/UR[j]) * np.array([[0, UR[j]*ER[j]], [-1*(UR[j]*ER[j]), 0]])
# C   = np.linalg.multi_dot([X, B, np.linalg.inv(A),X, B])
#D   = np.linalg.multi_dot([X, np.matmul(B, np.linalg.inv(A)), X, B])
#aTM  = np.array([1, 0,  0]) 
# Pr   = 1/erl * np.array([[0, url*erl], [-url*erl, 0]]) 
#Qr   = 1/url * np.array([[0, url*erl], [-1*(url*erl), 0]]) 
#E_perm = np.array([n_rid**2, n_def**2])
#H_perm = np.array([u_rid, u_def])
#ER = np.tile(E_perm, len(Lm)-1)                     # Array of permittivities in each layer  
#UR = np.tile(H_perm, len(Lm)-1)                     # Array of permeabilities in each layer
#ER = np.insert(ER, 0, n_ref_l**2)           
#UR = np.insert(UR, 0, u**2)           
'''
print(enumerate(Lm))

E_1 = [n_rid**2 for i in range(len(Lm)-1) if Lm.index(Lm[i])%2 !=0] 
E_2 = [n_def**2 for i in range(len(Lm)-1) if Lm.index(Lm[i])%2 == 0]

ER = [None]*(len(Ll)+len(Lr))
ER[::2] = E_1
ER[1::2] = E_2
print(len(Lm))
print(len(ER))
'''
'''
for i in range(len(L)):
    if enumerate(L[i])% == 0:
        L
'''
'''
S11 = np.dot(np.linalg.inv(A - C), np.linalg.multi_dot([X, B, np.linalg.inv(A), X, A]) - B)
S22 = S11
S12 = np.dot(np.dot(np.linalg.inv(A - D), X), A - np.linalg.multi_dot([B, np.linalg.inv(A), B]))
S21 = S12
'''
#for i in range(1,len(Lm)):
#    diff=Lm[i]-Lm[i-1]
#    print(diff)
