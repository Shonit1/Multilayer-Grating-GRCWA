import numpy as np
from config import *


pthick = [1.571,0.184,0.409]

# === Parameters ===
period = 0.76       # µm (grating period)
ridge_width_1 = 0.1625 # µm (50% duty cycle)
ridge_width_2 = 0.541

# Layer thicknesses (µm) for Case 2



def get_epgrids(Nx, Ny):
  # === Grid generation ===
  # Create x and y grids (Ny=1 still uses meshgrid)
  x0 = np.linspace(0, period, Nx, endpoint=False)
  y0 = np.linspace(0, L2[1], Ny, endpoint=False)  # dummy y (1 row)
  x, y = np.meshgrid(x0, y0, indexing='ij')   # shapes: (Nx, Ny)

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










'''
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
  '''

