import matplotlib.pyplot as plt
import numpy as np

from cahn_hilliard import CahnHilliardSolver
from velocity_fields import stokes_flow, constant_velocity, zero_velocity, divergence

from ui import get_animation, add_secondary_axis, rebin_velocity_field
from initial_conditions import InitialConditions

import constants as c

# Vælg startbetingelser
IC_U = InitialConditions()
IC_U.add_pyrenoid()

IC_P = InitialConditions()
IC_P.phi[:] = 0

# Definer velocity-function
velocity_function = stokes_flow
vel = velocity_function(IC_U.phi + IC_P.phi)

def late_stokes_flow(t, phi):
    if t < 5:
        return zero_velocity(phi)
    else:
        return vel

# Kør simulation
chs = CahnHilliardSolver(IC_U.phi, IC_P.phi, 100, 200, late_stokes_flow)
T, PHI_U, PHI_P, MEAN_PHI_U, MEAN_PHI_P = chs.run_simulation()

# Massebevarelse?
#plt.figure()
#plt.plot(T, MEAN_PHI, "o", label=r"Gennemsnit af $\phi$ over tid")
#plt.legend()

# Lav animation
fig, ax = plt.subplots(ncols=2, figsize=(10, 7))
anim1 = get_animation(T, PHI_U, fig, ax[0])
anim2 = get_animation(T, PHI_P, fig, ax[1])

# Tegn hastighedsfelt ved start
for axis in ax:
    axis.quiver(*rebin_velocity_field(IC_U.X, IC_U.Y, *vel), scale=np.max(vel), scale_units="xy")

plt.show()
#anim.save("animation.gif", writer="pillow", fps=20)