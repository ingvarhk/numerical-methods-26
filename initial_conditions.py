import constants as c
import numpy as np

class InitialConditions:
    def __init__(self):
        self.x = np.arange(0, c.lx, c.dx)
        self.y = np.arange(0, c.ly, c.dy)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.shape = np.shape(self.X)

        self.phi_equilibrium = np.sqrt(c.b/c.a)
        self.phi = np.zeros(self.shape) - self.phi_equilibrium # Stable background

    def add_random_bubbles(self):
        self.phi += np.random.random(self.shape)*1.5-1

    def add_pyrenoid(self, value: float = 1.0):
        R = np.sqrt((self.X - c.x0)**2 + (self.Y - c.y0)**2)
        self.phi[R < c.R0] = value

    def add_stardust(self, scale: float = 0.1):
        self.phi += (np.random.random(self.shape)-0.5)*scale # Star dust

    def get_phi(self):
        return self.phi