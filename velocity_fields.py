import numpy as np
import matplotlib.pyplot as plt

import constants as c
from initial_conditions import InitialConditions

IC = InitialConditions()

# Ligning: eta*laplace(v) - gradient(P) + epsilon*v = 0
def stokes_flow(phi):

    Nx = len(IC.x)
    Ny = len(IC.y)

    P = c.P0 * np.exp(-(IC.X - c.x0)**2 / (2 * c.sigma**2))
    #P[np.abs(IC.Y-c.x0) > 10] = 0

    kx = 2 * np.pi * np.fft.fftfreq(Nx, c.dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, c.dy)

    Kx, Ky = np.meshgrid(kx, ky)
    K_squared = Kx**2 + Ky**2

    vx_tilde = - 1j * Kx * np.fft.fft2(P) / (c.epsilon + K_squared * c.eta)
    vy_tilde = - 1j * Ky * np.fft.fft2(P) / (c.epsilon + K_squared * c.eta)

    vx = np.fft.ifft2(vx_tilde).real
    vy = np.fft.ifft2(vy_tilde).real

    return np.array([vx, vy]) # * (1 / np.sqrt(2 * np.pi * 6**2)) * np.exp(-(IC.Y - c.y0)**2 / (2 * 7**2))

def constant_velocity(phi):
    return np.array([
        np.ones_like(phi),      # vx = 1
        np.zeros_like(phi)      # vy = 0
    ])

def zero_velocity(phi):
    return np.array([
        np.zeros_like(phi),
        np.zeros_like(phi)
    ])

# Udregn divergens af hastighedsfelt
def divergence(v):
    _, Ny, Nx = v.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, c.dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, c.dy)

    Kx, Ky = np.meshgrid(kx, ky)
    K_mega = np.array([Kx, Ky])

    return np.fft.ifft2(np.sum(1j * K_mega * np.fft.fft2(v), axis=0)).real

# Plot
if __name__ == "__main__":
    v = stokes_flow(IC.phi)

    plt.title(r"Divergens af hastighedsfeltet, altså $\nabla\cdot v$")
    plt.imshow(divergence(v), extent=[0, c.lx, 0, c.ly])
    plt.colorbar()

    plt.figure()
    plt.title("Hastighedsfelt med gaussisk tryk")
    plt.quiver(IC.X, IC.Y, *v, scale=1.5*np.max(v), scale_units="xy")

    plt.show()