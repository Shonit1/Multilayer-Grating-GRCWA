import grcwa
import numpy as np
import matplotlib.pyplot as plt
from config import *
from layers import *
#from angle import diffraction_angles_um

def solver_system(nG,L1,L2,f,theta,phi,pthick,Nx,Ny,p,s):

    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose=0)


    obj.Add_LayerUniform(0.1,1)

    for layer_thickness in pthick:
        obj.Add_LayerGrid(layer_thickness,Nx,Ny)
        
    obj.Add_LayerUniform(0.366,2.1014)
   
    
    

    for i in range(13):
        obj.Add_LayerUniform(0.127, 4.41)
        obj.Add_LayerUniform(0.184, 2.1014)

      

    obj.Add_LayerUniform(0.127,4.41)    

    obj.Add_LayerUniform(0.1,2.25)

    obj.Init_Setup( )


    epgrid1, epgrid2, epgrid3 = get_epgrids(Nx, Ny)
    epgrid = np.concatenate((epgrid1.flatten(),epgrid2.flatten(),epgrid3.flatten()))
    obj.GridLayer_geteps(epgrid)
    

    planewave={'p_amp':p,'s_amp':s,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    

    Ri, Ti = obj.RT_Solve(normalize=1,byorder=1)
    
    return Ri, Ti, obj



Ri, Ti, obj = solver_system(nG,L1,L2,freq,theta,phi,pthick,Nx = 6,Ny=1,p=1,s=0)
print(obj.G[2])
print(Ri[2])
R = np.sum(Ri)
T = np.sum(Ti)
print(f"R: {R}, T: {T}, R+T: {R+T}")

#theta,phi,prop = diffraction_angles_um(obj.kx[2], obj.ky[2], 1/freq, n=1.0, region='reflected')
#print(f"Diffraction angle (degrees): theta = {theta}, phi = {phi}, prop = {prop}")


def Check_Nx():
    Nx_array = range(5,20)
    Ri_array = np.array([])
    for i in Nx_array:
        Ri, Ti, obj = solver_system(nG,L1,L2,freq,theta,phi,pthick,i,Ny=1,p=1,s=0)
        print(obj.G[1])
        if -1 in obj.G[1]:

            Ri_array = np.append(Ri_array,Ri[1])
        else:
            Ri_array = np.append(Ri_array,Ri[2])    
        print(f"Completed Nx: {i}")


    plt.plot(Nx_array, Ri_array, 'o-', label='P-polarization', linewidth=2)
    plt.xlabel('Number of x grid points (Nx)') 
    plt.ylabel('convergence of Ri')
    plt.title('Convergence of Reflectance with Increasing Nx')
    plt.legend()    
    plt.grid(True)
    plt.show()
    return Ri_array

#Check_Nx()



'''
def height_plot():
    p0 = np.linspace(0,1,20)
    q0 = np.linspace(0,1,20)
    t, q = np.meshgrid(p0, q0, indexing='ij')
    # initialize z as a 2D array
    z = np.zeros_like(t)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            x = t[i,j]
            y = q[i,j]
            print(x, y)
            pthick1 = [1,1,1.571]
            pthick1[0] = x
            pthick1[1] = y 
            Ri, Ti, obj = solver_system(nG,L1,L2,freq,theta,phi,pthick1,epgrid1,epgrid2,epgrid3,Nx,Ny,p=1,s=0)
            z[i, j] = Ri[2]
            print(Ri[2])
            print(obj.G[2])
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(t, q, z, shading='auto', cmap='viridis')
    plt.colorbar(label='z value')
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Heatmap of z = f(p, q)')
    plt.show()
    return


#height_plot()


'''




f_spectrum = np.linspace(1/end_wavelength, 1/start_wavelength, N_SAMP)
w_spectrum = 1 / f_spectrum

def array_calculator(p_array, s_array):
    p_array = np.array([])  
    s_array = np.array([])  
    
    for f in f_spectrum:
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,Nx=6,Ny=1,p=1,s=0)
        p_array = np.append(p_array,Ri[2])
    
    for f in f_spectrum:
        Ri, Ti, obj = solver_system(nG,L1,L2,f,theta,phi,pthick,Nx=6,Ny=1,p=0,s=1)      
        s_array = np.append(s_array,Ri[2])
        print(f"Completed frequency: {f}")

    print(obj.G[2])    
    return p_array, s_array 
p_array, s_array = array_calculator(np.array([]), np.array([]))

plt.plot(w_spectrum, p_array, label='P-polarization', linewidth=2)
plt.plot(w_spectrum, s_array, label='S-polarization', linewidth=2, linestyle='--')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Diffraction Efficiency -1st Order')
plt.title('-1st order optimized for P polarization-cycle duty - 0.57 and h2 = 0.184')
plt.legend()
plt.grid(True)
plt.show()


