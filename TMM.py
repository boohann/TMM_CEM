'''
This code calculates Transfer Matrix Method, which was described by Raymond 
C. Rumpf at the course Computational EM. I also used code of
Sathyanarayan Rao from https://www.mathworks.com/matlabcentral/fileexchange/47637-transmittance-and-reflectance-spectra-of-multilayered-dielectric-stack-using-transfer-matrix-method
Updated to Python and applied to repeating laser Bragg grating by Niall Boohan [niall.boohan@tyndall.ie]

NOTES:
-TE plane wave assumed no kx or ky components ###
'''
##############################################################################
# Dashboard
##############################################################################
### Import libraries ###
import scipy.linalg as la
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import norm

### Output arrays ###
Tx = []
Rx = []
Zx = []

### Device parameters ###
#lambda_0 =  1.5006             # Selected wavelength (um), Not nessesary 
n_rid = 3.2                     # Refractive index ridge
n_del = 0.005                   # Reflective index step
n_def = n_rid-n_del             # Refractive index etched defect
u_rid = 1                       # Magnetic perm layer 1
u_def = 1                       # Magnetic perm layer 2
WL_start = 1520                 # Start WL (nm)
WL_stop  = 1580                 # Stop WL (nm)
sw =0.1212                      # Slot width (um) lambda_0/(n_def*4)#1.1   
cl = 400                        # Cavity length
r_l = 0.5234                    # Left facet reflectivity
r_r = 0.5234#0.671              # Right facet reflectivity

### Calc to turn reflectivites to n ###
n_l = (n_rid*(1-r_l**2))/(1+r_l**2) 
n_r = (n_rid*(1-r_r**2))/(1+r_r**2)

### Load slot positions from external calculation file ###
L = np.load('Pslot.npy')
Lr = [L[i]-L[i-1]-sw/2 for i in range(1, len(L))]
Lr.insert(0 , L[0]-sw/2)
Ls = [sw for i in range(len(L))]
Lm = [None]*(len(Lr)+len(Ls))
Lm[1::2] = Ls
Lm[::2] = Lr
##############################################################################
##############################################################################
##############################################################################



##############################################################################
# Cavity calculations
##############################################################################
### Applying calculation matrices ###
I = np.identity(2)                                  # Creating identity matrix
lam0_nm  = np.linspace(WL_start, WL_stop, 1000)     # Free space wavelength 
lam0_m   = [i*1e-9 for i in lam0_nm]

''' Test QW DFB stack
uR = [u_rid,u_def]
eR = [ 1.69**2, 1.46**2]
L  = [ 0.1479, 0.1712]
ER = np.tile(eR, 15)                     # Array of permittivities in each layer  
UR = np.tile(uR, 15)
Lm = np.tile(L, 15)                     # Array of permeabilities in each layer
'''

U = [u_rid,u_def]
E = [n_rid**2, n_def**2]
ER = np.tile(E, len(L))                     # Array of permittivities in each layer  
UR = np.tile(U, len(L))

### Reflectivity arrays for left and right facet coatings ###
l_E = np.array([n_l**2])                     
l_H = np.array([1])                             
r_E = np.array([n_r**2])                    
r_H = np.array([1])                             

### Insert left & right facets & end section of laser ###
Lm.insert(0,0)
Lm.append(cl-L[-1]-sw/2)
Lm.append(0)
Lm = [i*1e-6 for i in Lm]           # Convert lengths to meters

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
print(ER)

### Generate input & output boudaries for laser [set to inf] ###
erl = 1.0                           # Left laser input
url = 1.0
err = 1.0                           # Right laser output
urr = 1.0
ni  = np.sqrt(erl)                  # Refrctive index of incidence media

### Source ###
pte      = 1/np.sqrt(2)
ptm      = pte*1j
##############################################################################
##############################################################################
##############################################################################



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
##############################################################################
##############################################################################



##############################################################################
# TMM calculation section
##############################################################################
### Run code for range of wavelengths ###
for i in range(len(lam0_m)):
   
    k0 =(2*np.pi)/lam0_m[i] 
    kz = np.sqrt(1)                             #Apply refractive index of incident material
    ### Remove reflectivity from phase change at facets ###
    Lm[0] = lam0_m[i]/(4*n_l)                   
    Lm[-1] = lam0_m[i]/(4*n_r)             
    print(Lm)
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
##############################################################################
##############################################################################



##############################################################################
# Visualisation
##############################################################################
### Plotting ###
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(lam0_nm, Rx, 'g-')
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
##############################################################################
##############################################################################
##############################################################################
