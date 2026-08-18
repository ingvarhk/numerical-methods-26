# SPATIAL DIMENSIONS
dy = dx = 0.5 # Spatial discretization step (0.07μm)
l = 40 # Domain size (5.7μm)

R0 = 7.2 # Initial pyrenoid radius (1.0μm)
x0 = l / 2 # Pyrenoid center (2.9μm)

# TIME
dt = 10e-3 # < 0.16 = 4*(kappa)/(lambda*b^2)


# NAVIER STOKES
eta = 40 # Shear viscosity (0.0002 Pa*min)
epsilon = 12.5 # Damping coefficient (0.0025 Pa*min*μm−2)

P0 = 10 # Hydrodynamic pressure (4*10−4 Pa)
sigma = 1.2 # Pressure standard deviation (5*10−5 Pa)


# CAHN-HILLIARD
lambd = 1 # Mobility coefficient (0.2 μm^2 min−1)
chi = 5 # Flory-Huggins interaction parameter (5)
kappa = 1 # Interfacial width coefficient (0.02 μm^2)

a = 1 # No equivalent in paper
b = chi # Suspected chi


# CONVERTING TO CHARACTERISTIC VALUES
def conversion(input: float, quantity: str) -> float:
    scale = 1

    if quantity == "length": scale = 0.14
    elif quantity == "time": scale = 0.01
    elif quantity == "pressure": scale = 4*10**(-5)
    else: print("You must specify quantity: Either 'length', 'time' or 'pressure'.")

    return input/scale