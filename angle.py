import grcwa
import numpy as np
import matplotlib.pyplot as plt
from config import *
from layers import *
from main import solver_system

f_spectrum = np.linspace(1/end_wavelength, 1/start_wavelength, N_SAMP)
w_spectrum = 1 / f_spectrum

import numpy as np

def diffraction_angles_um(kx, ky, wavelength_um, n=1.0, region='reflected'):
    """
    kx, ky : arrays (1/µm)
    wavelength_um : wavelength in µm
    n : refractive index (1 for air)
    region : 'reflected' or 'transmitted'
    Returns: theta_deg, phi_deg, propagating_mask
    """
    kx = np.array(kx, dtype=float)
    ky = np.array(ky, dtype=float)
    k0 = 2 * np.pi / wavelength_um  # 1/µm

    kz_sq = (n * k0)**2 - (kx**2 + ky**2)
    kz = np.sqrt(kz_sq + 0j)  # complex-safe sqrt

    if region == 'reflected':
        kz = -kz

    # Identify propagating orders
    propagating = np.real(kz_sq) > 0

    # Angles
    theta = np.full(kx.shape, np.nan)
    phi = np.degrees(np.arctan2(ky, kx))

    valid = propagating
    cos_theta = np.real(kz[valid]) / (n * k0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta[valid] = np.degrees(np.arccos(cos_theta))

    return theta, phi, propagating


theta_r_arr = np.array([])
phi_r_arr = np.array([])
prop_r_arr = np.array([])   

def angle_array(theta_r_arr, phi_r_arr, prop_r_arr):

    for f in f_spectrum:
        w = 1 / f
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p=1,s=0)

        kx = obj.kx[1]
        ky = obj.ky[1]
        theta_r, phi_r, prop_r = diffraction_angles_um(kx, ky, w, n=1.0, region='reflected')
        theta_r_arr = np.append(theta_r_arr, theta_r)
        phi_r_arr = np.append(phi_r_arr, phi_r)
        prop_r_arr = np.append(prop_r_arr, prop_r)

    return theta_r_arr, phi_r_arr, prop_r_arr


theta_r_arr, phi_r_arr, prop_r_arr = angle_array(theta_r_arr, phi_r_arr, prop_r_arr)

theta_rad = np.deg2rad(theta_r_arr)

y = -np.arcsin(w_spectrum/1.3)
y_shifted = y +3.5 

plt.plot(w_spectrum, theta_rad, 'o-',label='Diffracted Angle -1st Order', linewidth=2)
plt.plot(w_spectrum, y_shifted, 'r--', label='-Sine inverse wavelength')
plt.legend()
plt.xlabel('Wavelength (µm)')
plt.ylabel('Diffracted Angle (degrees)')
plt.title('Diffracted Angle vs Wavelength for -1st Order')
plt.grid()
plt.show()











'''

def order_angles(f):
    w = 1/f
    order_theta = np.array([])
    order_phi = np.array([])
    order_prop = np.array([])
    for i in range(nG-1):
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p=1,s=0)
        kx = obj.kx[i]
        ky = obj.ky[i]
        theta_r, phi_r, prop_r = diffraction_angles_um(kx, ky, w, n=1.0, region='reflected')
        print(theta_r, phi_r, prop_r)
        order_theta = np.append(order_theta, theta_r)
        order_phi = np.append(order_phi, phi_r)
        order_prop = np.append(order_prop, prop_r)
    return order_theta, order_phi, order_prop


order_theta, order_phi, order_prop = order_angles(1/1.064)
modes = range(nG-1)
plt.plot(modes, order_theta, 'o')
plt.xlabel('Diffracted Order')
plt.ylabel('Diffracted Angle (degrees)')
plt.title('Diffracted Angles for Each Order at 1.064 µm')
plt.grid()  
plt.show()

'''