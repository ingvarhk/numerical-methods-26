import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim

from constants import *

t = 0
t_max = 80

x = np.arange(0, l, dx); y = np.arange(0, l, dx)
N = len(x)

def s(X, Y, r):
    R = np.sqrt((X - np.mean(X))**2 + (Y - np.mean(Y))**2) # Distance from center
    return np.array(R < r, dtype=np.float64) # Mask

#phi = 2 * np.random.random((N, N)) - 1

X, Y = np.meshgrid(x, y)
R = np.sqrt((X - l/2)**2 + (Y - l/2)**2)

# Calculate the equilibrium phase value
#phi_eq = np.sqrt(b / a)

# Scale the initial profile to match the stable minima
#phi = 1 * np.tanh((R0 - R) / (np.sqrt(2 * kappa / b)))

phi = np.random.random((N, N))*1.5-1

phi_tilde = np.fft.fft2(phi)

k = 2*np.pi*np.fft.fftfreq(N, dx)
kx, ky = np.meshgrid(k, k)
k_squared = kx**2 + ky**2

phi_over_t = []
mean_values = []

i = 0
while t < t_max:
    phi = np.fft.ifft2(phi_tilde).real
    non_linear_term = np.fft.fft2(-a*phi**3 + b*phi)

    # Entire potential explicit
    phi_tilde = (phi_tilde + Lambda*k_squared*dt * non_linear_term) / (1 + Lambda*dt*k_squared*(k_squared*kappa))

    # Original (only c^3 explicit)
    #phi_tilde = (phi_tilde - Lambda*a*k_squared*non_linear_term*dt)/(1 + Lambda*dt*k_squared*(kappa*k_squared - b))

    if i % 100 == 0:
        mean_values.append([t, np.mean(phi)])
        phi_over_t.append([t, phi.copy()])
        
    t+=dt
    i+=1



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

    return anim.FuncAnimation(fig, update, frames=len(u_of_t), interval=20)

ani = get_animation(phi_over_t)

#plt.figure()
#plt.plot(*zip(*mean_values), "o")
plt.show()