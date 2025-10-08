import grcwa
import numpy as np
import matplotlib.pyplot as plt
from config import *
from layers import *

def solver_system(nG,L1,L2,f,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p,s):

    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose=1)


    obj.Add_LayerUniform(0.1,1)

    for layer_thickness in pthick:
        obj.Add_LayerGrid(layer_thickness,Nx,Ny)
        
    obj.Add_LayerUniform(0.3669,2.1014)
    obj.Add_LayerUniform(0.0001,1)

    obj.Add_LayerUniform(0.1267, 4.41)
    obj.Add_LayerUniform(0.1835, 2.1014)

    for i in range(12):
        obj.Add_LayerUniform(0.1267, 4.41)
        obj.Add_LayerUniform(0.1835, 2.1014)

      

    obj.Add_LayerUniform(0.1267,4.41)    

    obj.Add_LayerUniform(0.1,2.25)

    obj.Init_Setup()



    epgrid = np.concatenate((epgrid1.flatten(),epgrid2.flatten(),epgrid3.flatten()))
    obj.GridLayer_geteps(epgrid)

    planewave={'p_amp':p,'s_amp':s,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    

    Ri, Ti = obj.RT_Solve(normalize=1,byorder=1)
    
    return Ri, Ti, obj





f_spectrum = np.linspace(1/end_wavelength, 1/start_wavelength, N_SAMP)
w_spectrum = 1 / f_spectrum

def array_calculator(p_array, s_array):
    p_array = np.array([])  
    s_array = np.array([])  
    p=1
    s=0
    for f in f_spectrum:
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p,s)
        p_array = np.append(p_array,Ri[1])
    p=0
    s=1    
    for f in f_spectrum:
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p,s)      
        s_array = np.append(s_array,Ri[1])
        print(f"Completed frequency: {f}")

    print(obj.G[1])    
    return p_array, s_array 
p_array, s_array = array_calculator(np.array([]), np.array([]))

plt.plot(w_spectrum, p_array, label='P-polarization', linewidth=2)
plt.plot(w_spectrum, s_array, label='S-polarization', linewidth=2, linestyle='--')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Diffraction Efficiency')
plt.title('Diffraction Efficiency vs Wavelength')
plt.legend()
plt.grid(True)
plt.show()


