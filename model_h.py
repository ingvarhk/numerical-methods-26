import matplotlib.pyplot as plt
import numpy as np

from cahn_hilliard import cahn_hilliard
from velocity_fields import stokes_flow, constant_velocity, zero_velocity

from ui import get_animation, add_secondary_axis, rebin_velocity_field
from initial_conditions import InitialConditions

import constants as c

# Vælge startbetingelser
IC = InitialConditions()
IC.add_pyrenoid()

# Vælg velocity-funktion
velocity_function = constant_velocity

# Kør simulation
vel = velocity_function(IC.phi)
T, PHI, MEAN_PHI = cahn_hilliard(IC.phi, 200, 200, lambda _: vel)


# Lav animation
fig, ax = plt.subplots(figsize=(10, 7))
anim = get_animation(T, PHI, fig, ax)

# Tegn initial hastighedsfelt
vx, vy = velocity_function(IC.phi)
ax.quiver(*rebin_velocity_field(IC.X, IC.Y, vx, vy), scale=np.max(vx), scale_units="xy")

# Gammelt hastighedsplot
#ax2 = add_secondary_axis(ax)
#ax2.plot(x, vx[0], "--", color="red", lw=2)

plt.show()
#anim.save("animation.gif", writer="pillow", fps=20)