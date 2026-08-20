import matplotlib.pyplot as plt
import numpy as np

from cahn_hilliard import CahnHilliardSolver
from velocity_fields import stokes_flow, constant_velocity, zero_velocity, divergence

from ui import get_animation, add_secondary_axis, rebin_velocity_field
from initial_conditions import InitialConditions

import matplotlib.animation as anim

import constants as c

# Vælg startbetingelser
IC_U = InitialConditions()
IC_U.phi[:] = 0.1
IC_U.add_pyrenoid(0.8)

IC_P = InitialConditions()
IC_P.phi[:] = 0.1

print("Max:", np.max(IC_U.phi + IC_P.phi))

# Definer velocity-function
velocity_function = stokes_flow
vel = velocity_function(IC_U.phi + IC_P.phi)

def late_stokes_flow(t, phi):
    if t < 1000:
        return zero_velocity(phi)
    else:
        return vel

# Kør simulation
chs = CahnHilliardSolver(IC_U.phi, IC_P.phi, 50, 200, late_stokes_flow)
T, PHI_U, PHI_P, MEAN_PHI_U, MEAN_PHI_P = chs.run_simulation()

# save arrays
np.save("sim_data/T.npy", T)
np.save("sim_data/PHI_U.npy", PHI_U)
np.save("sim_data/PHI_P.npy", PHI_P)
np.save("sim_data/MEAN_PHI_U.npy", MEAN_PHI_U)
np.save("sim_data/MEAN_PHI_P.npy", MEAN_PHI_P)

# Massebevarelse?
plt.figure()
plt.plot(T, MEAN_PHI_U, "o", label=r"$\phi_U$")
plt.plot(T, MEAN_PHI_P, "o", label=r"$\phi_P$")
plt.plot(T, MEAN_PHI_U + MEAN_PHI_P, "o", label=r"$\phi_U + \phi_P$")
plt.legend()

# Lav animation
fig, ax = plt.subplots(ncols=2, figsize=(10, 7))

update_U = get_animation(T, PHI_U, fig, ax[0])
update_P = get_animation(T, PHI_P, fig, ax[1])

# Draw velocity field
for axis in ax:
    axis.quiver(
        *rebin_velocity_field(IC_U.X, IC_U.Y, *vel),
        scale=np.max(vel),
        scale_units="xy"
    )

def update(frame):
    return update_U(frame), update_P(frame)

combined_anim = anim.FuncAnimation(
    fig,
    update,
    frames=len(T),
    interval=5,
    blit=False
)

plt.show()

combined_anim.save(
    "please_virk2.gif",
    writer="pillow",
    fps=20
)