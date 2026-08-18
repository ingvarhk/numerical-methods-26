import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim

import constants as c
from velocity_fields import zero_velocity

def cahn_hilliard(phi0, t_max, samples, velocity=zero_velocity):
    
    SAMPLE_INTERVAL = int(t_max // (samples * c.dt))

    phi = phi0.copy()
    phi_tilde = np.fft.fft2(phi)

    Ny, Nx = phi.shape

    kx = 2 * np.pi * np.fft.fftfreq(Nx, c.dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, c.dy)

    Kx, Ky = np.meshgrid(kx, ky)
    k_squared = Kx**2 + Ky**2
    K_mega = np.array([Kx, Ky])

    T = np.zeros(samples)
    PHI = np.zeros((samples, Ny, Nx))
    MEAN_PHI = np.zeros(samples)

    t = 0

    T[0] = t
    PHI[0] = phi
    MEAN_PHI[0] = np.mean(phi)

    i = 1
    t += c.dt
    while t < t_max:
        velocity_term = np.fft.fft2(np.sum(velocity(phi) * np.fft.ifft2(1j * K_mega * phi_tilde), axis=0))

        # Entire potential explicit
        non_linear_term = np.fft.fft2(-c.a*phi**3 + c.b*phi)
        phi_tilde = (phi_tilde + c.lambd*k_squared*c.dt * non_linear_term - velocity_term * c.dt) / (1 + c.lambd*c.dt*k_squared*(k_squared*c.kappa))

        phi = np.fft.ifft2(phi_tilde).real

        # Original (only c^3 explicit)
        #phi_tilde = (phi_tilde - Lambda*a*k_squared*non_linear_term*dt)/(1 + Lambda*dt*k_squared*(kappa*k_squared - b))

        if i % SAMPLE_INTERVAL == 0:
            k = i // SAMPLE_INTERVAL

            T[k] = t
            PHI[k] = phi
            MEAN_PHI[k] = np.mean(phi)
            
        t += c.dt
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

if __name__ == "__main__":
    x = np.arange(0, c.l, c.dx)
    y = np.arange(0, c.l, c.dy)

    Nx = len(x)
    Ny = len(y)

    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - c.x0)**2 + (Y - c.x0)**2)

    # Equilibrium phase value
    #phi_eq = np.sqrt(b / a)

    phi = np.random.random((Ny, Nx))*1.5-1

    T, PHI, MEAN_PHI = cahn_hilliard(phi, 80, 100)
    ani = get_animation(T, PHI)

    #plt.figure()
    #plt.plot(*zip(*mean_values), "o")
    plt.show()