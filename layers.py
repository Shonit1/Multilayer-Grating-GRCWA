import numpy as np
from config import *


pthick = [0.409,0.150,1.571]

x0 = np.linspace(0,0.76,Nx,endpoint = False)
y0 = np.linspace(0,0.0001,Ny,endpoint = False)
x, y = np.meshgrid(x0,y0,indexing='ij')


#layer 1


# make epgrids: ridge where x < 0.38
ridge_width = 0.38  # µm
ridge_mask = (x < ridge_width)

epgrid1 = np.ones((Nx,Ny), dtype=complex) * 1.0   # default air
epgrid1[ridge_mask] = epS                        # top SiO2 in ridge

epgrid2 = np.ones((Nx,Ny), dtype=complex) * 1.0
epgrid2[ridge_mask] = epA                        # middle Al2O3 in ridge

epgrid3 = np.ones((Nx,Ny), dtype=complex) * 1.0
epgrid3[ridge_mask] = epS   
# make epgrids: ridge where x < 0.38
  