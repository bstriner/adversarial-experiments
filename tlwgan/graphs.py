import os

import matplotlib.pyplot as plt
import numpy as np


def makedir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def write_images(path, images):
    makedir(path)
    fig = plt.figure()
    im = np.transpose(images, (0, 2, 1, 3)).reshape((10 * 28, 10 * 28))
    plt.imshow(im, cmap='gray')
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from keras.datasets import mnist

    x = mnist.load_data()[0][0]
    print "Max: {}".format(np.max(x))
    x = x.astype(np.float32) / 255.0
    ind = np.random.randint(0, x.shape[0], (100,))
    x = x[ind, :].reshape((10, 10, 28, 28))
    write_images('output/test.png', x)
