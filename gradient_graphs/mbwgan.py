from keras.layers import Input, Dense, LeakyReLU, Lambda, Flatten, Add, Activation
from constraints import LInfConstraint
from keras.models import Model
from keras.optimizers import RMSprop
import theano
import theano.tensor as T
import itertools
import numpy as np


def mbwgan_loss(ytrue, ypred):
    return T.mean(-ytrue * ypred, axis=None)


def mbwgan_gradients(nb_batch, lr, hidden_dim,  con=lambda: None, reg=lambda: None, noise=1e-4):
    return lambda xreal, xfake: mbwgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, con, reg, noise)


def mbwgan_gradients_calc(xreal, xfake, nb_batch, lr, hidden_dim, con, reg, noise):
    n = xreal.shape[0]
    assert n == xfake.shape[0]
    xreal = np.expand_dims(xreal, 0)
    xfake = np.expand_dims(xfake, 0)
    print ("MBWGAN: {}, {}".format(xreal.shape, xfake.shape))

    input_shape = xreal.shape[1:]
    xr = Input(input_shape)
    xf = Input(input_shape)

    #act = lambda: LeakyReLU(0.2)
    act = lambda: Activation('tanh')
    h1 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    h2 = Dense(hidden_dim, kernel_constraint=con(), kernel_regularizer=reg())
    #h3 = Dense(hidden_dim, kernel_constraint=con())
    hy = Dense(1, kernel_constraint=con(), kernel_regularizer=reg())

    def f(x):
        perms = []
        for perm in itertools.permutations(range(n), n):
            xx = Lambda(lambda _x: T.concatenate(list(_x[:, p:p + 1, :] for p in perm), axis=1),
                        output_shape=lambda z: z)(x)
            xx = Flatten()(xx)
            perms.append(xx)

        def g(xxx):
            h = h1(xxx)
            h = act()(h)
            h = h2(h)
            #h = act()(h)
            #h = h3(h)
            h = act()(h)
            y = hy(h)
            return y

        ys = [g(xx) for xx in perms]
        ytot = Add()(ys)
        return ytot

    yr = f(xr)
    yf = f(xf)
    m = Model([xr, xf], [yr, yf])
    opt = RMSprop(lr)
    m.compile(opt, mbwgan_loss)
    targets = [np.ones((1, 1)), -1 * np.ones((1, 1))]
    for batch in range(nb_batch):
        _xr = xreal + np.random.normal(0, noise, xreal.shape)
        _xf = xfake + np.random.normal(0, noise, xfake.shape)
        loss = m.train_on_batch([_xr, _xf], targets)
        print "Batch: {}, Loss: {}".format(batch, loss[0])

    #grads, _ = theano.scan(lambda _i, _y, _x: T.grad(T.mean(_y, axis=None), _x)[0, _i, :],
    #                       sequences=[T.arange(xf.shape[1])], non_sequences=[yf, xf], outputs_info=[None])
    #grad_f = theano.function([xf], [grads])
    grads = T.grad(T.mean(yf, axis=None), xf)[0,:,:]
    grad_f = theano.function([xf], [grads])

    xgrads = grad_f(xfake)[0]
    print "Xgrads: {}".format(xgrads.shape)
    return xgrads
