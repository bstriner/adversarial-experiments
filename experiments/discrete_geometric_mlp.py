
import theano
import theano.tensor as T
from keras.initializations import zero, glorot_uniform
from keras.layers import Dense, Input
import keras.backend as K
from keras.optimizers import SGD
import numpy as np
from geometric_model import GeometricModel
from utils import geometric, plot_geometric
from tqdm import tqdm

def main():
    k = 3
    model = GeometricModel(k)
    xreal = geometric(k)
    nb_epoch = 2000
    nb_batches = 20
    nb_discriminator_batches = 5
    path = "discrete_geometric_mlp/{}".format(k)
    for epoch in tqdm(range(nb_epoch)):
        pngpath = "{}/epoch-{:04d}.png".format(path, epoch)
        plot_geometric(pngpath, xreal, model)
        for _ in range(nb_batches):
            for _ in range(nb_discriminator_batches):
                model.train_discriminator(xreal)
            model.train_generator()
            #model.train(xreal)


if __name__=="__main__":
    main()