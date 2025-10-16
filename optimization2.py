import grcwa
import numpy as np
import matplotlib.pyplot as plt
from config import *
from layers import *
import grcwa, nlopt
from autograd import grad
import autograd.numpy as np

grcwa.set_backend('autograd')

def get_epgrids1(t,Nx, Ny):
  # === Grid generation ===
  # Create x and y grids (Ny=1 still uses meshgrid)
  x0 = np.linspace(0, L1[0], Nx, endpoint=False)
  y0 = np.linspace(0, L2[1], Ny, endpoint=False)  # dummy y (1 row)
  x, y = np.meshgrid(x0, y0, indexing='ij')   # shapes: (Nx, Ny)
  ridge_width_1 = L1[0]/2 - t/2
  ridge_width_2 = L1[0]/2 + t/2
  # === Ridge definition ===
  ridge_mask = (x > ridge_width_1) & (x < ridge_width_2)  # boolean mask for ridge region

  # === Layer 1: Top SiO2 ===
  epgrid1 = np.ones((Nx, Ny), dtype=complex) * 1.0   # background = air
  epgrid1[ridge_mask] = epS

  # === Layer 2: Middle Al2O3 ===
  epgrid2 = np.ones((Nx, Ny), dtype=complex) * 1.0
  epgrid2[ridge_mask] = epA

  # === Layer 3: Bottom SiO2 ===
  epgrid3 = np.ones((Nx, Ny), dtype=complex) * 1.0
  epgrid3[ridge_mask] = epS

  # === Collect all layers in a list ===
  

  return epgrid1, epgrid2, epgrid3

def objective_function(x,nG,L1,L2,f,theta,phi,Nx,Ny,p,s,Qabs= np.inf):
    t,alt = x[0], x[1]
    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose=0)

    obj.Add_LayerUniform(0.1,1)
    pthick = [1.571,alt,0.409]

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


    epgrid1, epgrid2, epgrid3 = get_epgrids1(t,Nx, Ny)
    epgrid = np.concatenate((epgrid1.flatten(),epgrid2.flatten(),epgrid3.flatten()))
    obj.GridLayer_geteps(epgrid)
    

    planewave={'p_amp':p,'s_amp':s,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    

    Ri, Ti = obj.RT_Solve(normalize=1,byorder=1)
    
    if -1 in obj.G[1]:
        return Ri[1]
    else:
        return Ri[2]
    

ctrl = 0
Qabs = np.inf
fun = lambda x: objective_function(x,nG,L1,L2,f = freq,theta =theta,phi = phi,Nx=6,Ny=1,p=0,s=1,Qabs = np.inf)

grad_fun = grad(fun)





def fun_nlopt(x,gradn):
    global ctrl

    gradn[:] = grad_fun(x)
    y = fun(x)

    print('Step = ',ctrl,', R = ',y)
    ctrl += 1
    return fun(x)



#setup NLOPT

opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([0,0])   # duty cycle and height limits
opt.set_upper_bounds([0.76,0.5])
opt.set_maxeval(50)
opt.set_xtol_rel(1e-4)
opt.set_max_objective(fun_nlopt)



x0 = 0.5*np.random.random(2)  # initial guess

x_opt = opt.optimize(x0)
print("Optimized parameters:", x_opt)

#Ri = objective_function([0.38,0.150],nG,L1,L2,f = freq,theta =theta,phi = phi,Nx=6,Ny=1,p=1,s=0,Qabs = np.inf)
#print("Reflectance at optimized parameters:", Ri)