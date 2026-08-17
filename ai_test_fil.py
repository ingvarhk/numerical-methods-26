import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim

# Constants
R0 = 7.2
Lambda = 1.0
kappa = 1.0 
a = 1.0
b = 5.0

# Domain & Discretization
l = 40.0
dx = 0.5
dt = 1e-2

x = np.arange(0, l, dx)
y = np.arange(0, l, dx)
N = len(x)

X, Y = np.meshgrid(x, y)

# Smooth initial profile using hyperbolic tangent instead of sharp mask
R = np.sqrt((X - l/2)**2 + (Y - l/2)**2)
phi = np.tanh((R0 - R) / (np.sqrt(2 * kappa / b)))

# Wavenumbers
k = 2 * np.pi * np.fft.fftfreq(N, dx)
kx, ky = np.meshgrid(k, k)
k_squared = kx**2 + ky**2

phi_tilde = np.fft.fft2(phi)

t = 0
t_max = 50
phi_over_t = []

# Stabilization parameter (S >= 3 * a * phi_max^2)
S = 15.0

# Precompute 2/3 dealiasing mask
k_max = np.max(np.abs(k))
dealias_mask = (np.abs(kx) < (2 / 3) * k_max) & (np.abs(ky) < (2 / 3) * k_max)

# Precompute stable implicit denominator
denom = 1 + Lambda * dt * k_squared * (kappa * k_squared + S)

i = 0
mean_values = []
while t < t_max:
  phi = np.fft.ifft2(phi_tilde).real

  # Explicit evaluation of non-linear + destabilizing terms
  explicit_space = a * (phi**3) - (b + S) * phi
  explicit_fourier = np.fft.fft2(explicit_space) * dealias_mask

  # Stabilized semi-implicit update
  phi_tilde = (phi_tilde - Lambda * dt * k_squared * explicit_fourier) / denom

  if i % 100 == 0:
    mean_values.append([t, np.mean(phi)])
    phi_over_t.append([t, phi.copy()])
  t += dt
  i += 1


# Function returning playable animation
def get_animation(u_of_t):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(u_of_t[0][1], vmax=3, vmin=-3)
    fig.colorbar(im, ax=ax)

    ax.set(xlabel="x", ylabel="y")

    def update(frame):
        im.set_data(u_of_t[frame][1])
        #im.set_clim(vmax=np.max(u_of_t[frame][1]), vmin=np.min(u_of_t[frame][1]))
        ax.set_title(f"t = {u_of_t[frame][0]:.2f}")

        return im

    return anim.FuncAnimation(fig, update, frames=len(u_of_t), interval=50)

ani = get_animation(phi_over_t)

plt.figure()
plt.plot(*zip(*mean_values), "o", label="Mean")
plt.show()