import numpy as np



L1 = [0.76,0]
L2 = [0.0,0.0001]

nG = 10

freq = 1/1.064

theta = 0.7732

phi = 0
N_SAMP = 50
CENTER_WAV = 1.064 

WIDTH = 0.3

start_wavelength = CENTER_WAV - WIDTH
end_wavelength = CENTER_WAV + WIDTH

N_SAMP = 20 # number of frequency samples for plotting



# array of target wavelengths
f_sampled = np.array([1 / CENTER_WAV])

Nx,Ny = 300,300

epS = 2.1014
epA = 2.7254
epT = 4.41
