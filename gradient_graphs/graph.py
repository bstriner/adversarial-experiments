import matplotlib.pyplot as plt
import os
import numpy as np


def graph_gradients(path, xreal, xfake, xfake_gradients):
    if isinstance(xfake_gradients, tuple):
        discriminator = xfake_gradients[1]
        xfake_gradients = xfake_gradients[0]
    else:
        discriminator = None
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    fig = plt.figure()
    if discriminator:
        ls = np.linspace(-2, 2, 50)
        gs = np.meshgrid(ls, ls)
        i1 = gs[0].reshape((-1, 1))
        i2 = gs[1].reshape((-1, 1))
        idx = np.hstack((i1, i2))
        y = discriminator(idx)
        img = np.flip(y.reshape((50, 50)), 0)
        plt.imshow(img, extent=(-2, 2, -2, 2), cmap='coolwarm')

    plt.scatter(xreal[:, 0], xreal[:, 1], c='g', s=400)
    plt.scatter(xfake[:, 0], xfake[:, 1], c='r', s=400)
    n = xfake.shape[0]
    #xend = xfake + xfake_gradients
    for i in range(n):
        scale = 0.4 / np.sqrt(np.sum(np.square(xfake_gradients[i,:])))
        dx, dy = xfake_gradients[i, 0]*scale, xfake_gradients[i, 1]*scale
        plt.arrow(xfake[i, 0], xfake[i, 1], dx, dy,
                  head_width=0.05, head_length=0.1)
        # fc=fc, ec=ec, alpha=alpha, width=width, head_width=head_width,
        # head_length=head_length, **arrow_params)
    lim = 2
    plt.ylim(-lim, lim)
    plt.xlim(-lim, lim)
    fig.savefig(path)
    plt.close(fig)


def write_gradients(path, xfake, xfake_gradients):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if isinstance(xfake_gradients, tuple):
        xfake_gradients=xfake_gradients[0]
    n = xfake.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("Point: <{},{}>, Gradient: <{},{}>\n".format(xfake[i, 0], xfake[i, 1],
                                                                 xfake_gradients[i, 0],
                                                                 xfake_gradients[i, 1]))
