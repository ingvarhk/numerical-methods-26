from matplotlib import pyplot as plt
import matplotlib.animation as anim

import constants as c

# Function returning playable animation
def get_animation(t, phi, fig, ax):

    im = ax.imshow(phi[0], extent=[0, c.lx, 0, c.ly], origin="lower")
    fig.colorbar(im, ax=ax)

    ax.set(xlabel="x", ylabel="y")

    def update(frame):
        im.set_data(phi[frame])
        #im.set_clim(vmax=np.max(u_of_t[frame][1]), vmin=np.min(u_of_t[frame][1]))
        ax.set_title(f"t = {t[frame]:.1f}")

        return im

    # return anim.FuncAnimation(fig, update, frames=len(phi), interval=5)
    return update

# Tilføjer en venstre akse
def add_secondary_axis(ax):
    ax2 = ax.twinx()

    ax2.yaxis.tick_left()
    ax2.tick_params(axis="y", colors="red")

    #ax2.set(ylim=(-0.15, 0.15))
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set(position=("outward", 45), color="red")

    return ax2

def rebin_velocity_field(X, Y, vx, vy):
    n = 2
    ny, nx = vx.shape
    ny2, nx2 = ny // n * n, nx // n * n

    vx = vx[:ny2, :nx2].reshape(ny2//n, n, nx2//n, n).mean((1, 3))
    vy = vy[:ny2, :nx2].reshape(ny2//n, n, nx2//n, n).mean((1, 3))

    Xq = X[n//2:ny2:n, n//2:nx2:n]
    Yq = Y[n//2:ny2:n, n//2:nx2:n]

    return Xq, Yq, vx, vy
