import matplotlib.pyplot as plt
import numpy as np

from cahn_hilliard import cahn_hilliard
from velocity_fields import stokes_flow, constant_velocity, zero_velocity, divergence

from ui import get_animation, add_secondary_axis, rebin_velocity_field
from initial_conditions import InitialConditions

import constants as c

# Vælg startbetingelser
IC = InitialConditions()
IC.add_pyrenoid(IC.phi_equilibrium * 1.5)

# Definer velocity-function
velocity_function = stokes_flow
vel = velocity_function(IC.phi)

# Kør simulation
T, PHI, MEAN_PHI = cahn_hilliard(IC.phi, 500, 200, lambda _: vel)

# Massebevarelse?
plt.figure()
plt.plot(T, MEAN_PHI, "o", label=r"Gennemsnit af $\phi$ over tid")
plt.legend()

# Lav animation
fig, ax = plt.subplots(figsize=(10, 7))
anim = get_animation(T, PHI, fig, ax)

# Tegn hastighedsfelt ved start
vx, vy = velocity_function(IC.phi)
ax.quiver(*rebin_velocity_field(IC.X, IC.Y, vx, vy), scale=np.max(vx), scale_units="xy")

plt.show()
#anim.save("animation.gif", writer="pillow", fps=20)