import os

os.environ["THEANO_FLAGS"] = "optimizer=None,device=gpu"
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam
import theano
import theano.tensor as T

import numpy as np
import keras.backend as K
import itertools
from keras.layers import Lambda


def wgan_loss(ytrue, ypred):
    return -ytrue * ypred


def tlwgan_gradients(nb_batch, lr, hidden_dim, noise=1e-4):
    return lambda xreal, xfake: tlwgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, noise)


def tlwgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, noise):
    n = xreal.shape[0]
    assert n == xfake.shape[0]
    input_dim = xreal.shape[1]
    xr = Input((input_dim,))
    xf = Input((input_dim,))
    act = lambda: Activation('relu')
    h1 = Dense(hidden_dim)
    hy = Dense(1)

    def f(x):
        h = h1(x)
        h = act()(h)
        y = hy(h)
        return y

    yr = f(xr)
    yf = f(xf)

    W = h1.kernel
    U = hy.kernel
    mask_params = []
    l1s = []
    for i, combo in enumerate(itertools.product([-1, 1], repeat=input_dim)):
        mask = K.variable(np.random.random((hidden_dim,)), dtype='float32', name="mask_{}".format(i))
        wu = T.dot(W * (mask.dimshuffle(('x', 0))), U)
        l1 = 0
        for j, c in enumerate(combo):
            l1 += wu[j, 0] * c
        mask_params.append(mask)
        l1s.append(l1)
    l1s = T.concatenate([T.reshape(_x, (1,)) for _x in l1s])
    # masks = T.concatenate(T.reshape(_x, (1, -1)) for _x in mask_params)

    l1loss = -T.sum(l1s, axis=0)
    l1opt = Adam(1e-3)
    l1con = lambda _x: T.clip(_x, 0, 1)
    l1cons = {k: l1con for k in mask_params}
    l1updates = l1opt.get_updates(mask_params, l1cons, l1loss)
    l1max = T.max(l1s, axis=0)
    l1fun = theano.function([], [l1loss, l1max], updates=l1updates)
    rescale = Lambda(lambda _x: _x / l1max, output_shape=lambda _x: _x)

    yf = rescale(yf)
    yr = rescale(yr)

    m = Model([xr, xf], [yr, yf])
    opt = Adam(lr)
    m.compile(opt, wgan_loss)
    targets = [np.ones((n, 1)), -1 * np.ones((n, 1))]

    for epoch in range(nb_batch):
        _xr = xreal + np.random.normal(0, noise, xreal.shape)
        _xf = xfake + np.random.normal(0, noise, xfake.shape)
        for i in range(64):
            _l1loss, _l1max = l1fun()
            #print "L1loss: {}, l1: {}".format(_l1loss, _l1max)
        loss = m.train_on_batch([_xr, _xf], targets)
        print "TLWgan epoch: {}, loss: {}".format(epoch, loss[0])

    grads, _ = theano.scan(lambda _i, _y, _x: T.grad(_y[_i, 0], _x)[_i, :],
                           sequences=[T.arange(xf.shape[0])], non_sequences=[yf, xf])
    grad_f = theano.function([xf], [grads])

    discriminator = Model([xf], [yf])
    xgrads = grad_f(xfake)[0]
    weights = {"W": K.get_value(h1.kernel), "b": K.get_value(h1.bias)}
    return xgrads, lambda _x: discriminator.predict(_x), weights
