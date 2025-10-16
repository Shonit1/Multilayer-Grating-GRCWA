import grcwa
import numpy as np
import matplotlib.pyplot as plt
from config import *
from layers import *
from main import solver_system









def check_Convergence():
    NG = range(100)
    Ri_array = np.array([])
    for i in NG:
        Ri, Ti, obj = solver_system(i+20,L1,L2,freq,theta,phi,pthick,epgrid1,epgrid2,epgrid3,Nx,Ny,p=1,s=0)
        print(obj.G[1])
        if -1 in obj.G[1]:

            Ri_array = np.append(Ri_array,Ri[1])
        else:
            Ri_array = np.append(Ri_array,Ri[2])    
        print(f"Completed nG: {i+1}")




    plt.plot(NG, Ri_array, 'o-', label='P-polarization', linewidth=2)
    plt.xlabel('Number of Fourier Orders (nG)') 
    plt.ylabel('convergence of Ri')
    plt.title('Convergence of Reflectance with Increasing Fourier Orders')
    plt.legend()    
    plt.grid(True)
    plt.show()
    return Ri_array