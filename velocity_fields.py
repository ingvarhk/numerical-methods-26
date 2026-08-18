import numpy as np
import matplotlib.pyplot as plt

import constants as c

def stokes_flow(phi):
    x = np.arange(0, c.lx, c.dx)
    y = np.arange(0, c.ly, c.dy)

    Nx = len(x)
    Ny = len(y)

    X, Y = np.meshgrid(x, y)

    P = c.P0 * np.exp(-(X - c.x0)**2 / (2 * c.sigma**2))

    kx = 2 * np.pi * np.fft.fftfreq(Nx, c.dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, c.dy)

    Kx, Ky = np.meshgrid(kx, ky)
    K_squared = Kx**2 + Ky**2


    vx_tilde = - 1j * Kx * np.fft.fft2(P) / (c.epsilon + K_squared * c.eta)

    vy_tilde = - 1j * Ky * np.fft.fft2(P) / (c.epsilon + K_squared * c.eta)

    vx = np.fft.ifft2(vx_tilde).real
    vy = np.fft.ifft2(vy_tilde).real

    # x, y, X, Y, P
    return np.array([vx, vy])

def constant_velocity(phi):
    return np.array([
        np.ones_like(phi),      # u = 1
        np.zeros_like(phi)      # v = 0
    ])

def zero_velocity(phi):
    return np.array([
        np.zeros_like(phi),      # u = 1
        np.zeros_like(phi)      # v = 0
    ])

# x, y, X, Y, P, vx, vy = stokes_flow()

# fig, ax = plt.subplots(ncols=2)
# imx = ax[0].imshow(vx, origin="lower")
# imy = ax[1].imshow(vy, origin="lower")
# fig.colorbar(imx)
# fig.colorbar(imy)

# plt.figure()
# plt.quiver(X, Y, vx, vy)

# plt.figure()
# plt.plot(x, P[0])

# plt.figure()
# plt.plot(x, vx[0])

# plt.show()