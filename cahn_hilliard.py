import numpy as np
from matplotlib import pyplot as plt

import constants as c
from velocity_fields import zero_velocity

import time

class CahnHilliardSolver:
    def __init__(self, phi0_u, phi0_p, t_max, samples, velocity=zero_velocity, print_progress=True):
        self.velocity = velocity
        self.print_progress = print_progress
        self.samples = samples

        self.iterations = int(t_max / c.dt)
        self.SAMPLE_INTERVAL = self.iterations // samples

        self.phi_u = phi0_u.copy()
        self.phi_p = phi0_p.copy()

        self.phi_u_tilde = np.fft.fft2(self.phi_u)
        self.phi_p_tilde = np.fft.fft2(self.phi_p)

        self.Ny, self.Nx = self.phi_u.shape

        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, c.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, c.dy)

        self.Kx, self.Ky = np.meshgrid(kx, ky)
        self.k_squared = self.Kx**2 + self.Ky**2
        self.K_mega = np.array([self.Kx, self.Ky])

        self.T = np.zeros(samples)
        self.PHI_U = np.zeros((samples, self.Ny, self.Nx))
        self.PHI_P = np.zeros((samples, self.Ny, self.Nx))
        self.MEAN_PHI_U = np.zeros(samples)
        self.MEAN_PHI_P = np.zeros(samples)

        self.t = 0

        self.T[0] = self.t
        self.PHI_U[0] = self.phi_u
        self.PHI_P[0] = self.phi_p
        self.MEAN_PHI_U[0] = np.mean(self.phi_u)
        self.MEAN_PHI_P[0] = np.mean(self.phi_p)

        self.start_time = time.time()

    # Reaction
    def reaction_flux(self):
        sigma = 1 - 1/(1+np.exp(-c.sharpness_b*(self.phi_u-c.phi_c)))
        return -sigma*c.k_p*self.phi_u + (1-sigma)*c.k_u*self.phi_p

    # Update function
    def spectral_ch_step(self, phi, phi_tilde, reaction_direction):
        velocity_term = np.sum(1j * self.K_mega*np.fft.fft2(self.velocity(self.t, phi) * phi), axis=0)
        
        # Entire potential explicit
        if reaction_direction == 1: non_linear_term = np.fft.fft2(c.a*phi**3 - c.b*phi)
        else: non_linear_term = np.fft.fft2(8.3*phi) # Define this later...

        reaction_flux = np.fft.fft2(reaction_direction*self.reaction_flux())
        phi_tilde = (phi_tilde - c.lambd*self.k_squared*c.dt * non_linear_term - velocity_term*c.dt + reaction_flux*c.dt) / (1 + c.lambd*c.dt*self.k_squared*(self.k_squared*c.kappa))

        phi = np.fft.ifft2(phi_tilde).real

        return phi, phi_tilde

    # Cahn-Hilliard
    def run_simulation(self):
        for i in range(1, self.iterations):
            self.t = i * c.dt

            temp_phi_u, temp_phi_u_tilde = self.spectral_ch_step(self.phi_u, self.phi_u_tilde, 1)
            self.phi_p, self.phi_p_tilde = self.spectral_ch_step(self.phi_p, self.phi_p_tilde, -1)

            self.phi_u, self.phi_u_tilde = temp_phi_u, temp_phi_u_tilde
            
            if i % self.SAMPLE_INTERVAL == 0:
                k = i // self.SAMPLE_INTERVAL
                if k >= self.samples: break

                self.T[k] = self.t
                self.PHI_U[k] = self.phi_u
                self.PHI_P[k] = self.phi_p

                self.MEAN_PHI_U[k] = np.mean(self.phi_u)
                self.MEAN_PHI_P[k] = np.mean(self.phi_p)

                # Printing how far we have come
                if self.print_progress: print(f"\rStatus: {100*i/self.iterations:.1f}%		Remaining: ~{(time.time()-self.start_time)*(self.iterations/i - 1):.0f}s ", end="", flush=True)

        print(f"\rStatus: 100.0%		Remaining: ~0s ", flush=True)
        print(f"Done! Completed in {time.time()-self.start_time:.1f}s.")

        return self.T, self.PHI_U, self.PHI_P, self.MEAN_PHI_U, self.MEAN_PHI_P


    def central_drop(X, Y, r):
        R = np.sqrt((X - np.mean(X))**2 + (Y - np.mean(Y))**2) # Distance from center
        return np.array(R < r, dtype=np.float64) # Mask

if __name__ == "__main__":
    from ui import get_animation

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

    fig, ax = plt.subplots(figsize=(10, 7))
    ani = get_animation(T, PHI, fig, ax)

    #plt.figure()
    #plt.plot(*zip(*mean_values), "o")
    plt.show()