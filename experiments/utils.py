import numpy as np
import matplotlib.pyplot as plt
import os


def geometric(k):
    i = np.arange(k)
    angles = (i * 2 * np.pi / k).reshape((-1, 1))
    verts = np.hstack((np.cos(angles), np.sin(angles)))
    return verts

def geometric_gmm(k, n):
    i = np.arange(k)
    angles = (i * 2 * np.pi / k).reshape((-1, 1))
    verts = np.hstack((np.cos(angles), np.sin(angles)))





def plot_geometric(path, xreal, model, limit=2, n=100):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    linspace = np.linspace(-limit, limit, n)
    grid = np.meshgrid(linspace, linspace)
    grid = [np.expand_dims(z, 2) for z in grid]
    points = np.concatenate(grid, axis=2)
    xsamples = points.reshape((-1, 2))
    ysamples = model.discriminator(xsamples)
    yimg = ysamples.reshape((n,n))
    xgen = model.get_generator()

    fig = plt.figure()
    # hot, seismic, coolwarm
    plt.imshow(yimg, cmap='coolwarm', extent=(-limit, limit, -limit, limit))
    plt.scatter(xreal[:, 0], xreal[:, 1], c="r", s=200)
    plt.scatter(xgen[:, 0], xgen[:, 1], c="b", s=200)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    fig.savefig(path)
    plt.close(fig)
