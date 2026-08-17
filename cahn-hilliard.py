import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim

# Constants
R0 = 7.2 # 1 mikrometer

Lambda = 1
kappa = 1 
a = 1 # Log difference ~ c^3
b = 5 # Suspected chi

# Simulation
l = 40 # domain size
dx = 0.5
dt = 10e-3 # < 0.16 = 4*(kappa)/(Lambda*b^2)

t = 0
t_max = 20

def F(u):
    return u**3

x = np.arange(0, l, dx); y = np.arange(0, l, dx)
N = len(x)

def s(X, Y, r):
    R = np.sqrt((X - np.mean(X))**2 + (Y - np.mean(Y))**2) # Distance from center
    return np.array(R < r, dtype=np.float64) # Mask

#phi = 2 * np.random.random((N, N)) - 1

X, Y = np.meshgrid(x, y)
phi = s(X, Y, R0)
phi[phi < 0.9] = -1

phi_tilde = np.fft.fft2(phi)

k = 2*np.pi*np.fft.fftfreq(N, dx)
kx, ky = np.meshgrid(k, k)
k_squared = kx**2 + ky**2

phi_over_t = []
mean_values = []

i = 0
while t < t_max:
    phi = np.fft.ifft2(phi_tilde).real
    non_linear_term = np.fft.fft2(F(phi))

    # Entire potential explicit
    #phi_tilde = (phi_tilde + Lambda*k_squared*dt * (-a*non_linear_term + b*phi_tilde)) / (1 + Lambda*dt*k_squared*(k_squared*kappa))

    # Original (only c^3 explicit)
    phi_tilde = (phi_tilde - Lambda*a*k_squared*non_linear_term*dt)/(1 + Lambda*dt*k_squared*(kappa*k_squared - b))

    if i%25 == 0:
        mean_values.append([t, np.mean(phi)])
        phi_over_t.append([t, phi.copy()])
        
    t+=dt
    i+=1



# Function returning playable animation
def get_animation(u_of_t):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(u_of_t[0][1], vmax=1, vmin=-1)
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