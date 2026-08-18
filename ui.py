from matplotlib import pyplot as plt
import matplotlib.animation as anim

import constants as c

# Function returning playable animation
def get_animation(t, phi, fig, ax):

    im = ax.imshow(phi[0], vmax=3, vmin=-3, extent=[0, c.l, 0, c.l], origin="lower")
    fig.colorbar(im, ax=ax)

    ax.set(xlabel="x", ylabel="y")

    def update(frame):
        im.set_data(phi[frame])
        #im.set_clim(vmax=np.max(u_of_t[frame][1]), vmin=np.min(u_of_t[frame][1]))
        ax.set_title(f"t = {t[frame]:.1f}")

        return im

    return anim.FuncAnimation(fig, update, frames=len(phi), interval=5)

# Tilføjer en venstre akse
def add_secondary_axis(ax):
    ax2 = ax.twinx()

    ax2.yaxis.tick_left()
    ax2.tick_params(axis="y", colors="red")

    #ax2.set(ylim=(-0.15, 0.15))
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set(position=("outward", 45), color="red")

    return ax2