import keras.backend as K
import numpy as np
import theano.tensor as T
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm


def clip_constraint(_x):
    return T.clip(_x, 0, 1)


class TLWGan(object):
    def __init__(self, input_dim, latent_dim, hidden_dim,
                 mask_count, opt_g, opt_d, opt_l):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.mask_count = mask_count
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.opt_l = opt_l
        self.srng = RandomStreams(123)
        self.generator = self.generator_model()
        self.discriminator, self.kernels = self.discriminator_model()
        self.train_l_fun, self.reset_fun, self.save_fun, self.lmax = self.lipschitz_model()

        input_x = Input((input_dim,), name='input_x')
        input_z = Input((latent_dim,), name='input_z')
        rescale = Lambda(lambda _x: _x / self.lmax, output_shape=lambda _x: _x)

        xfake = self.generator(input_z)
        yfake = self.discriminator(xfake)
        yreal = self.discriminator(input_x)
        yfake = rescale(yfake)
        yreal = rescale(yreal)

        dloss = T.sum(yfake, axis=None) - T.sum(yreal, axis=None)
        dupdates = self.opt_d.get_updates(self.discriminator.trainable_weights, {}, dloss)
        self.train_d_fun = K.function([input_x, input_z], [dloss], updates=dupdates)

        gloss = -T.sum(yfake, axis=None)
        gupdates = self.opt_g.get_updates(self.generator.trainable_weights, {}, gloss)
        self.train_g_fun = K.function([input_z], [gloss], updates=gupdates)

    def generator_model(self):
        input_z = Input((self.latent_dim,), name='generator_input_z')
        d1 = Dense(self.hidden_dim, activation='relu')
        d2 = Dense(self.hidden_dim, activation='relu')
        dy = Dense(self.input_dim, activation='sigmoid')
        y = dy(d2(d1(input_z)))
        m = Model([input_z], [y])
        return m

    def discriminator_model(self):
        input_x = Input((self.input_dim,), name='discriminator_input_x')
        d1 = Dense(self.hidden_dim, activation='relu')
        d2 = Dense(self.hidden_dim, activation='relu')
        dy = Dense(1)
        y = dy(d2(d1(input_x)))
        m = Model([input_x], [y])
        kernels = [d1.kernel, d2.kernel, dy.kernel]
        return m, kernels

    def make_mask(self, name):
        return K.variable(np.random.random((self.hidden_dim,)), dtype='float32', name=name)

    def lipschitz_for_mask(self, wmask, umask):
        w, u, v = self.kernels
        wm = w * (wmask.dimshuffle(('x', 0)))
        um = u * (umask.dimshuffle(('x', 0)))
        wuv = T.dot(T.dot(wm, um), v)
        return T.sum(T.abs_(wuv), axis=0)

    def mask_reset_value(self):
        return self.srng.uniform(low=0, high=1, size=(self.hidden_dim,))

    def lipschitz_model(self):
        wmask1 = self.make_mask('wmask')
        umask1 = self.make_mask('umask')
        wmasks = [wmask1]
        umasks = [umask1]
        l1 = self.lipschitz_for_mask(wmask1, umask1)
        ls = [l1]
        resets = []
        for i in range(self.mask_count):
            wmask = self.make_mask("wmask_{}".format(i))
            umask = self.make_mask("umask_{}".format(i))
            l = self.lipschitz_for_mask(wmask, umask)
            wmasks.append(wmask)
            umasks.append(umask)
            ls.append(l)
            resets.append(K.update(wmask, self.mask_reset_value()))
            resets.append(K.update(umask, self.mask_reset_value()))

        ls = T.concatenate([T.reshape(_x, (1,)) for _x in ls], axis=0)
        wmaskv = T.concatenate([T.reshape(_x, (1, -1)) for _x in wmasks], axis=0)
        umaskv = T.concatenate([T.reshape(_x, (1, -1)) for _x in umasks], axis=0)
        lmax = T.max(ls, axis=None)
        loss = -T.sum(ls, axis=None)

        params = wmasks + umasks
        constraints = {p: clip_constraint for p in params}
        updates = self.opt_l.get_updates(params, constraints, loss)
        train_fun = K.function([], [loss], updates=updates)
        reset_fun = K.function([], [], updates=resets)

        argmax = T.argmax(ls, axis=0)
        wmax = wmaskv[argmax, :]
        umax = umaskv[argmax, :]
        supdates = [K.update(wmask1, wmax), K.update(umask1, umax)]
        save_fun = K.function([], [], updates=supdates)
        return train_fun, reset_fun, save_fun, lmax

    def fit(self, gen, epochs, batches, batches_d, batches_l, callback=None):
        for epoch in tqdm(range(epochs), "Training"):
            dloss = []
            gloss = []
            lloss = []
            for _ in tqdm(range(batches), "Epoch: {}".format(epoch)):
                for _ in range(batches_d):
                    self.save_fun([])
                    self.reset_fun([])
                    for _ in range(batches_l):
                        lloss.append(self.train_l_fun([]))
                    x, z = next(gen)
                    dloss.append(self.train_d_fun([x, z]))
                x, z = next(gen)
                gloss.append(self.train_g_fun([z]))
            dloss = np.mean(dloss)
            gloss = np.mean(gloss)
            lloss = np.mean(lloss)
            tqdm.write("Epoch: {}, G Loss: {}, D Loss: {}, L Loss: {}".format(epoch, gloss, dloss, lloss))
            if callback:
                callback(epoch)
