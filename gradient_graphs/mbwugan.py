from keras.layers import Input, Dense, LeakyReLU, Lambda, Flatten, merge
from constraints import LInfConstraint
from keras.models import Model
from keras.optimizers import RMSprop, SGD
import theano
import theano.tensor as T
import itertools
import numpy as np
from tqdm import tqdm


def mbwugan_loss(ytrue, ypred):
    return -ytrue * ypred


def mbwugan_gradients(nb_pretrain_batch, nb_unroll_batch, lr, hidden_dim, con):
    return lambda xreal, xfake: mbwugan_gradients_calc(xreal, xfake,
                                                       nb_pretrain_batch, nb_unroll_batch, lr, hidden_dim, con)


def mbwugan_gradients_calc(xreal, xfake, nb_pretrain_batch, nb_unroll_batch, lr, hidden_dim, con):
    n = xreal.shape[0]
    assert n == xfake.shape[0]
    print ("MBWUGAN: {}, {}".format(xreal.shape, xfake.shape))

    input_dim = np.prod(xreal.shape)
    xr = T.fmatrix("xr")
    xf = T.fmatrix("xf")

    act = lambda x: T.nnet.relu(x, 0.2)
    h1 = Dense(hidden_dim, kernel_constraint=con())
    h2 = Dense(hidden_dim, kernel_constraint=con())
    h3 = Dense(hidden_dim, kernel_constraint=con())
    hy = Dense(1, kernel_constraint=con())
    h1.build((None, input_dim))
    h2.build((None, hidden_dim))
    h3.build((None, hidden_dim))
    hy.build((None, hidden_dim))
    layers = [h1, h2, h3, hy]
    Ws = [l.kernel for l in layers]
    bs = [l.bias for l in layers]
    params = Ws + bs

    def f(x, _params):
        perms = []
        for perm in itertools.permutations(range(n), n):
            xx = T.concatenate(list(x[p:p + 1, :] for p in perm), axis=1)
            xx = T.reshape(xx, (1, -1))
            perms.append(xx)

        def g(xxx):
            h = act(T.dot(xxx, _params[0]) + _params[4])
            h = act(T.dot(h, _params[1]) + _params[5])
            h = act(T.dot(h, _params[2]) + _params[6])
            y = T.dot(h, _params[3]) + _params[7]
            return y

        ys = [g(xx) for xx in perms]
        ytot = sum(ys)
        return ytot

    def calc_updates(_xreal, _xfake, _params, _params_t):
        yr = f(_xreal, _params)
        yf = f(_xfake, _params)
        loss = T.mean(yf, axis=None) - T.mean(yr, axis=None)
        newparams = [pt - lr * T.grad(loss, p) for p, pt in zip(_params, _params_t)]
        for i, l in enumerate(layers):
            newparams[i] = l.kernel_constraint(newparams[i])
        return newparams

    # pretrain
    pretrain_f = theano.function([xr, xf], [],
                                 updates=[(p, newp) for p, newp in zip(params, calc_updates(xr, xf, params, params))])
    for _ in tqdm(range(nb_pretrain_batch), desc="Pretraining MBWUGAN"):
        pretrain_f(xreal, xfake)

    # unrolled loss
    params_t = params
    for _ in tqdm(range(nb_unroll_batch), desc="Unrolling MBWUGAN"):
        params_t = calc_updates(xr, xf, params, params_t)

    gloss = T.mean(f(xf, params_t), axis=None)

    grad = T.grad(gloss, xf)
    grad_f = theano.function([xr, xf], [grad])
    xgrads = grad_f(xreal, xfake)[0]
    return xgrads
