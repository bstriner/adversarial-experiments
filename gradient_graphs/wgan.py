from keras.layers import Input, Dense, LeakyReLU, Activation
from constraints import LInfConstraint
from keras.models import Model
from keras.optimizers import RMSprop, Adam
import theano
import theano.tensor as T

import numpy as np


def wgan_loss(ytrue, ypred):
    return -ytrue * ypred


def wgan_gradients(nb_batch, lr, hidden_dim, con=lambda: None, reg=lambda:None, noise=1e-4):
    return lambda xreal, xfake: wgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, con, reg, noise)


def wgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, con, reg, noise):
    n = xreal.shape[0]
    assert n == xfake.shape[0]
    input_dim = xreal.shape[1]
    xr = Input((input_dim,))
    xf = Input((input_dim,))
    #act = lambda: LeakyReLU(0.2)
    act = lambda : Activation('tanh')
    h1 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    h2 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    h3 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    hy = Dense(1, kernel_constraint=con(), kernel_regularizer=reg())

    def f(x):
        h = h1(x)
        h = act()(h)
        h = h2(h)
        h = act()(h)
        h = h3(h)
        h = act()(h)
        y = hy(h)
        return y

    yr = f(xr)
    yf = f(xf)
    m = Model([xr, xf], [yr, yf])
    opt = Adam(lr)
    m.compile(opt, wgan_loss)
    targets = [np.ones((n, 1)), -1 * np.ones((n, 1))]
    for epoch in range(nb_batch):
        _xr = xreal + np.random.normal(0, noise, xreal.shape)
        _xf = xfake + np.random.normal(0, noise, xfake.shape)
        loss = m.train_on_batch([_xr, _xf], targets)
        print "Wgan epoch: {}, loss: {}".format(epoch, loss[0])

    grads, _ = theano.scan(lambda _i, _y, _x: T.grad(_y[_i, 0], _x)[_i, :],
                           sequences=[T.arange(xf.shape[0])], non_sequences=[yf, xf])
    grad_f = theano.function([xf], [grads])

    discriminator = Model([xf], [yf])
    xgrads = grad_f(xfake)[0]
    return xgrads, lambda _x: discriminator.predict(_x)
