from cahn_hilliard import cahn_hilliard, get_animation
from velocity_fields import stokes_flow, constant_velocity
import matplotlib.pyplot as plt
import numpy as np
import constants as c

x = np.arange(0, c.l, c.dx)
y = np.arange(0, c.l, c.dy)

Nx = len(x)
Ny = len(y)
phi = np.random.random((Ny, Nx))*1.5-1

X, Y = np.meshgrid(x, y)
R = np.sqrt((X - c.x0)**2 + (Y - c.x0)**2)

phi = (R < c.R0)*1.5-1

T, PHI, MEAN_PHI = cahn_hilliard((R < c.R0)*1.5-1, 200, 100, stokes_flow)

ani = get_animation(T, PHI)
plt.show()