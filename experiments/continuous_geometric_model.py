import theano
import theano.tensor as T
from keras.initializations import zero, glorot_uniform
from keras.layers import Dense, Input
import keras.backend as K
from keras.optimizers import SGD
import numpy as np


class ContinuousGeometricModel(object):
    def __init__(self, k, m):
        self.k = k

        z = Input((m,))
        generator_h = Dense(64, activation="sigmoid")
        generator_x = Dense(2)
        generator = generator_x(generator_h(z))

        self.generator = generator

        x = Input((2,))
        discriminator_h = Dense(64, activation="sigmoid")
        discriminator_y = Dense(1, activation="sigmoid")
        realy = discriminator_y(discriminator_h(x))
        fakey = discriminator_y.call(discriminator_h.call(generator))

        discriminator_loss = K.sum(K.log(fakey)) - K.sum(K.log(realy))
        generator_loss = -K.sum(K.log(fakey))

        discriminator_params = discriminator_h.weights + discriminator_y.weights
        generator_params = generator_h.weights+generator_x.weights

        self.discriminator_opt = SGD(1e-3)
        self.discriminator_updates = self.discriminator_opt.get_updates(discriminator_params, [], discriminator_loss)
        self.generator_opt = SGD(1e-4)
        self.generator_updates = self.generator_opt.get_updates(generator_params, [], generator_loss)
        self.updates = self.discriminator_updates + self.generator_updates
        self.train_function = K.function([x, z], [discriminator_loss, generator_loss], updates=self.updates)
        self.train_discriminator_function = K.function([x, z], [], updates=self.discriminator_updates)
        self.train_generator_function = K.function([z], [], updates=self.generator_updates)
        self.discriminator_function = K.function([x], [realy])
        self.generator_function = K.function([z], [xgen])

    def train(self, x, z):
        return self.train_function([x, z])

    def train_generator(self, z):
        self.train_generator_function([z])

    def train_discriminator(self, x, z):
        self.train_discriminator_function([x, z])

    def discriminator(self, x):
        return self.discriminator_function([x])[0]

    def get_generator(self, z):
        return self.generator_function([z])[0]
