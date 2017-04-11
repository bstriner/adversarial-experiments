import theano
import theano.tensor as T
from keras.initializations import zero, glorot_uniform
from keras.layers import Dense, Input
import keras.backend as K
from keras.optimizers import SGD
import numpy as np


class GeometricModel(object):
    def __init__(self, k):
        self.k = k
        generator = glorot_uniform((k, 2))
        self.generator = generator

        x = Input((2,))
        discriminator_h = Dense(64, activation="sigmoid")
        discriminator_y = Dense(1, activation="sigmoid")
        realy = discriminator_y(discriminator_h(x))
        fakey = discriminator_y.call(discriminator_h.call(generator))

        discriminator_loss = K.sum(K.log(fakey)) - K.sum(K.log(realy))
        generator_loss = -K.sum(K.log(fakey))

        discriminator_params = discriminator_h.weights + discriminator_y.weights
        generator_params = [generator]

        self.discriminator_opt = SGD(3e-3)
        self.discriminator_updates = self.discriminator_opt.get_updates(discriminator_params, [], discriminator_loss)
        self.generator_opt = SGD(1e-3)
        self.generator_updates = self.generator_opt.get_updates(generator_params, [], generator_loss)
        self.updates = self.discriminator_updates + self.generator_updates
        self.train_function = K.function([x], [discriminator_loss, generator_loss], updates=self.updates)
        self.train_discriminator_function = K.function([x], [], updates=self.discriminator_updates)
        self.train_generator_function = K.function([], [], updates=self.generator_updates)
        self.discriminator_function = K.function([x], [realy])

    def train(self, x):
        return self.train_function([x])

    def train_generator(self):
        self.train_generator_function([])

    def train_discriminator(self, x):
        self.train_discriminator_function([x])

    def discriminator(self, x):
        return self.discriminator_function([x])[0]

    def get_generator(self):
        return K.get_value(self.generator)
