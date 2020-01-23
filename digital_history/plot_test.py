import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython import embed

def pl():
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative


    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)

    # points[0] = (x[0], y[0]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    # segments[0] = 2x2 matrix. segments[0][0] = points[0]; segments[0][1] = points[1]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1)

    # Create a continuous norm to map from data points to colors
    # min(norm) = 0, max(norm) = 1
    norm = plt.Normalize(dydx.min(), dydx.max())
    # cmap = color map
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)


    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(-1.1, 1.1)


    embed()

    plt.show()

if __name__ == '__main__':
    pl()