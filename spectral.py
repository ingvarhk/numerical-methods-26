import numpy as np
import matplotlib.pyplot as plt

dx = 0.5
dy = 0.5
l = 40

x = np.arange(0, l, dx)
y = np.arange(0, l, dy)

Nx = len(x)
Ny = len(y)

X, Y = np.meshgrid(x, y)

epsilon = 12.5
eta = 40

x0 = l / 2 # 20
sigma = 1.2
P0 = 10
P = P0 * np.exp(-(x - x0)**2 / (2 * sigma**2))

P, _ = np.meshgrid(P, P)

kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

Kx, Ky = np.meshgrid(kx, ky)
K_squared = Kx**2 + Ky**2


vx_tilde = - 1j * Kx * np.fft.fft2(P) / (epsilon + K_squared * eta)

vy_tilde = - 1j * Ky * np.fft.fft2(P) / (epsilon + K_squared * eta)

vx = np.fft.ifft2(vx_tilde).real
vy = np.fft.ifft2(vy_tilde).real

fig, ax = plt.subplots(ncols=2)
imx = ax[0].imshow(vx, origin="lower")
imy = ax[1].imshow(vy, origin="lower")
fig.colorbar(imx)
fig.colorbar(imy)

plt.figure()
plt.quiver(X, Y, vx, vy)

plt.figure()
plt.plot(x, P[0])

plt.figure()
plt.plot(x, vx[0])

plt.show()