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
dy = 0.5
dt = 10e-3 # < 0.16 = 4*(kappa)/(Lambda*b^2)

def cahn_hilliard(phi0, dt, t_max, samples):
    n_steps = int(round(t_max / dt))
    sample_rate = max(1, n_steps // (samples - 1))

    phi = phi0.copy()
    phi_tilde = np.fft.fft2(phi)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)
    kx, ky = np.meshgrid(kx, ky)
    k_squared = kx**2 + ky**2

    # phi_over_t = []
    # mean_values = []

    T = np.zeros(samples)
    PHI = np.zeros(samples)
    MEAN_PHI = np.zeros(samples)

    t = 0

    T[0] = t
    PHI[0] = phi
    MEAN_PHI[0] = np.mean(phi)

    i = 1
    k = 1
    t += dt
    while t < t_max:
        phi = np.fft.ifft2(phi_tilde).real
        non_linear_term = np.fft.fft2(-a*phi**3 + b*phi)

        # Entire potential explicit
        phi_tilde = (phi_tilde + Lambda*k_squared*dt * non_linear_term) / (1 + Lambda*dt*k_squared*(k_squared*kappa))

        # Original (only c^3 explicit)
        #phi_tilde = (phi_tilde - Lambda*a*k_squared*non_linear_term*dt)/(1 + Lambda*dt*k_squared*(kappa*k_squared - b))

        if i % sample_rate == 0:
            # mean_values.append([t, np.mean(phi)])
            # phi_over_t.append([t, phi.copy()])
            T[k] = t
            PHI[k] = phi
            MEAN_PHI[k] = np.mean(phi)
            k += 1
            
        t += dt
        i += 1

    return T, PHI, MEAN_PHI
    # return phi_over_t, mean_values



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

T, PHI, MEAN_PHI = cahn_hilliard(R < 0.5, dt, 80, 100)
ani = get_animation(np.array(T, PHI))

#plt.figure()
#plt.plot(*zip(*mean_values), "o")
plt.show()