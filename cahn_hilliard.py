import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim

from constants import *

def cahn_hilliard(phi0, dt, t_max, samples):

    SAMPLE_INTERVAL = int(t_max // (samples * dt))

    phi = phi0.copy()
    phi_tilde = np.fft.fft2(phi)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)

    kx, ky = np.meshgrid(kx, ky)
    k_squared = kx**2 + ky**2

    T = np.zeros(samples)
    PHI = np.zeros((samples, Ny, Nx))
    MEAN_PHI = np.zeros(samples)

    t = 0

    T[0] = t
    PHI[0] = phi
    MEAN_PHI[0] = np.mean(phi)

    i = 1
    t += dt
    while t < t_max:
        phi = np.fft.ifft2(phi_tilde).real

        # Entire potential explicit
        non_linear_term = np.fft.fft2(-a*phi**3 + b*phi)
        phi_tilde = (phi_tilde + lambd*k_squared*dt * non_linear_term) / (1 + lambd*dt*k_squared*(k_squared*kappa))

        # Original (only c^3 explicit)
        #phi_tilde = (phi_tilde - Lambda*a*k_squared*non_linear_term*dt)/(1 + Lambda*dt*k_squared*(kappa*k_squared - b))

        if i % SAMPLE_INTERVAL == 0:
            k = i // SAMPLE_INTERVAL
            
            T[k] = t
            PHI[k] = phi
            MEAN_PHI[k] = np.mean(phi)
            
        t += dt
        i += 1

    return T, PHI, MEAN_PHI


# Function returning playable animation
def get_animation(t, phi):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(phi[0], vmax=3, vmin=-3)
    fig.colorbar(im, ax=ax)

    ax.set(xlabel="x", ylabel="y")

    def update(frame):
        im.set_data(phi[frame])
        #im.set_clim(vmax=np.max(u_of_t[frame][1]), vmin=np.min(u_of_t[frame][1]))
        ax.set_title(f"t = {t[frame]:.2f}")

        return im

    return anim.FuncAnimation(fig, update, frames=len(phi), interval=20)


def central_drop(X, Y, r):
    R = np.sqrt((X - np.mean(X))**2 + (Y - np.mean(Y))**2) # Distance from center
    return np.array(R < r, dtype=np.float64) # Mask

x = np.arange(0, l, dx)
y = np.arange(0, l, dy)

Nx = len(x)
Ny = len(y)

X, Y = np.meshgrid(x, y)
R = np.sqrt((X - l / 2)**2 + (Y - l / 2)**2)

# Equilibrium phase value
#phi_eq = np.sqrt(b / a)

phi = np.random.random((Ny, Nx))*1.5-1

T, PHI, MEAN_PHI = cahn_hilliard(phi, dt, 80, 100)
ani = get_animation(T, PHI)

#plt.figure()
#plt.plot(*zip(*mean_values), "o")
plt.show()