import numpy as np



L1 = [0.76,0]
L2 = [0.0,0.1]

nG = 5

freq = 1/1.064

theta = np.pi/180 * 44.9

phi = 0
N_SAMP = 20
CENTER_WAV = 1.064

WIDTH = 0.005

start_wavelength = CENTER_WAV - WIDTH
end_wavelength = CENTER_WAV + WIDTH





# array of target wavelengths
f_sampled = np.array([1 / CENTER_WAV])

#Nx,Ny = 15,1

epS = 2.1014
epA = 2.7254
epT = 4.41
