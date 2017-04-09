import os

os.environ["THEANO_FLAGS"] = "optimizer=None,device=gpu"
from keras.layers import Input, Dense, LeakyReLU, Activation
from constraints import LInfConstraint
from keras.models import Model
from keras.optimizers import RMSprop, Adam
import theano
import theano.tensor as T

import numpy as np
import keras.backend as K
import itertools
from keras.layers import Lambda
def wgan_loss(ytrue, ypred):
    return -ytrue * ypred


def wgan_gradients(nb_batch, lr, hidden_dim, con=lambda: None, reg=lambda:None, noise=1e-4, norm=False, ltest=False):
    return lambda xreal, xfake: wgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim,
                                                    con, reg, noise, norm, ltest)


def wgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, con, reg, noise, norm, ltest):
    n = xreal.shape[0]
    assert n == xfake.shape[0]
    input_dim = xreal.shape[1]
    xr = Input((input_dim,))
    xf = Input((input_dim,))
    #act = lambda: LeakyReLU(0.2)
    #act = lambda : Activation('tanh')
    act = lambda : Activation('relu')
    h1 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    #h2 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    #h3 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    hy = Dense(1, kernel_constraint=con(), kernel_regularizer=reg())


    def f(x):
        h = h1(x)
        h = act()(h)
        #h = h2(h)
        #h = act()(h)
        #h = h3(h)
        #h = act()(h)
        y = hy(h)
        return y

    yr = f(xr)
    yf = f(xf)
    if ltest:
        l1s = []
        for combo in itertools.product([0, 1], repeat=hidden_dim):
            wp = h1.kernel * (np.array(combo).reshape((1, -1)))
            wu = T.dot(wp, hy.kernel)
            l1 = T.reshape(T.sum(T.abs_(wu), axis=None), (1,))
            l1s.append(l1)
        l1s = T.concatenate(l1s, axis=0)
        maxl1 = T.max(l1s, axis=0)
        yr = Lambda(lambda _y: _y/maxl1, output_shape=lambda _y: _y)(yr)
        yf = Lambda(lambda _y: _y/maxl1, output_shape=lambda _y: _y)(yf)
    m = Model([xr, xf], [yr, yf])
    opt = Adam(lr)
    m.compile(opt, wgan_loss)
    targets = [np.ones((n, 1)), -1 * np.ones((n, 1))]

    mean_shift = np.zeros((1, 2), dtype=np.float32)
    if norm:
        mean_shift = np.mean(xreal, axis=0, keepdims=True)-np.mean(xfake, axis=0, keepdims=True)

    for epoch in range(nb_batch):
        _xr = xreal + np.random.normal(0, noise, xreal.shape)
        _xf = xfake + mean_shift+np.random.normal(0, noise, xfake.shape)
        loss = m.train_on_batch([_xr, _xf], targets)
        print "Wgan epoch: {}, loss: {}".format(epoch, loss[0])

    grads, _ = theano.scan(lambda _i, _y, _x: T.grad(_y[_i, 0], _x)[_i, :],
                           sequences=[T.arange(xf.shape[0])], non_sequences=[yf, xf])
    grad_f = theano.function([xf], [grads])

    discriminator = Model([xf], [yf])
    xgrads = grad_f(xfake)[0]
    weights = {"W":K.get_value(h1.kernel), "b":K.get_value(h1.bias)}
    return xgrads, lambda _x: discriminator.predict(_x+mean_shift), weights
