import matplotlib.pyplot as plt
import numpy as np

from cahn_hilliard import cahn_hilliard
from velocity_fields import stokes_flow, constant_velocity, zero_velocity

from ui import get_animation, add_secondary_axis, rebin_velocity_field
import constants as c

x = np.arange(0, c.lx, c.dx)
y = np.arange(0, c.ly, c.dy)
X, Y = np.meshgrid(x, y)

phi = np.random.random((len(x), len(y)))*1.5-1

R = np.sqrt((X - c.x0)**2 + (Y - c.y0)**2)
phi = (R < c.R0)*1.0

phi[phi > 0.5] = 1
phi[phi < 0.5] = -np.sqrt(c.b/c.a)

velocity_function = constant_velocity

# Simulation
vel = stokes_flow(phi)
T, PHI, MEAN_PHI = cahn_hilliard(phi, 200, 200, lambda _: vel)

print(PHI.shape)

# Animation
fig, ax = plt.subplots(figsize=(10, 7))
anim = get_animation(T, PHI, fig, ax)

# Plot hastighed ved start
vx, vy = velocity_function(phi)
ax.quiver(*rebin_velocity_field(X, Y, vx, vy), scale=np.max(vx), scale_units="xy")

# Gammelt hastighedsplot
#ax2 = add_secondary_axis(ax)
#ax2.plot(x, vx[0], "--", color="red", lw=2)

plt.show()
anim.save("animation.gif", writer="pillow", fps=20)