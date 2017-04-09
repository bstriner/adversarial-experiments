import os

os.environ["THEANO_FLAGS"] = "optimizer=None,device=gpu"
import numpy as np
import theano.tensor as T
import theano
from keras.optimizers import Adam
import itertools
import keras.backend as K


def solve_numpy(W, U):
    k = W.shape[1]
    assert (U.shape[0] == k)
    maxl1 = 0
    maxcombo = None
    for combo in itertools.product([0, 1], repeat=k):
        Wp = W * np.array(combo).reshape((1, -1))
        l1 = np.sum(np.abs(np.dot(Wp, U)), axis=None)
        if l1 > maxl1:
            maxl1 = l1
            maxcombo = combo
    print "Max L1: {}, {}".format(maxl1, maxcombo)
    return np.array(maxcombo) > 0.5


def solve_T(W, U):
    k = W.shape[1]
    assert (U.shape[0] == k)
    # init_mask = np.random.random((k,))
    init_mask = np.zeros((k,))
    # init_mask = np.ones((k,))
    mask = K.variable(init_mask, dtype='float32', name="mask")
    w = theano.shared(W, name="W")
    u = theano.shared(U, name="U")
    wp = w * (mask.dimshuffle(('x', 0)))
    wu = T.dot(wp, u)
    l1 = T.sum(T.abs_(wu), axis=None)
    loss = -l1
    params = [mask]
    constraints = {mask: lambda _m: T.clip(_m, 0, 1)}
    opt = Adam(1e-3)
    updates = opt.get_updates(params, constraints, loss)
    fun = theano.function([], [l1], updates=updates)
    for i in range(1024):
        _l1 = fun()[0]
        # print "Opt L1: {}".format(_l1)
    _mask = mask.get_value()
    print "Mask {}, {}".format(_l1, _mask)

    m = _mask > 0.5
    Wp = W * (m.reshape((1, -1)))
    WU = np.dot(Wp, U)
    _l1 = np.sum(np.abs(WU))
    print "L1 test: {}, {}".format(_l1, m)
    return m


def solve_T2(W, U):
    input_dim = W.shape[0]
    k = W.shape[1]
    assert U.shape[0] == k
    output_dim = U.shape[1]
    assert output_dim == 1
    w = theano.shared(W, name="W")
    u = theano.shared(U, name="U")
    masks = []
    l1s = []
    for i, combo in enumerate(itertools.product([-1, 1], repeat=input_dim)):
        init_mask = np.zeros((k,))
        mask = K.variable(init_mask, dtype='float32', name="mask_{}".format(i))
        wp = w * (mask.dimshuffle(('x', 0)))
        wu = T.dot(wp, u)
        l1 = 0
        for j, c in enumerate(combo):
            l1 += wu[j, 0] * c
        masks.append(mask)
        l1s.append(l1)

    l1s = T.concatenate([T.reshape(_x, (1,)) for _x in l1s], axis=0)
    loss = -T.sum(l1s, axis=None)
    opt = Adam(1e-3)
    con = lambda _m: T.clip(_m, 0, 1)
    params = masks
    cons = {p: con for p in params}
    updates = opt.get_updates(params, cons, loss)
    fun = theano.function([], [loss], updates=updates)

    l1 = T.max(l1s, axis=0)
    argmax = T.argmax(l1s, axis=0)
    masks = T.concatenate([T.reshape(_x, (1, -1)) for _x in masks], axis=0)
    maskmax = masks[argmax, :]
    pred_fun = theano.function([], [l1, maskmax])

    for i in range(2048):
        loss = fun()[0]
        _l1, _ = pred_fun()
        # print "Loss: {}, L1: {}".format(loss, _l1)
    _l1, _mask = pred_fun()
    print "Mask {}, {}".format(_l1, _mask)

    m = _mask > 0.5
    Wp = W * (m.reshape((1, -1)))
    WU = np.dot(Wp, U)
    _l1 = np.sum(np.abs(WU))
    print "L1 test: {}, {}".format(_l1, m)
    return m


def experiment(W, U):
    # print "W {}".format(W.shape)
    # print W
    # print "U {}".format(U.shape)
    # print U
    m1 = solve_numpy(W, U)
    m2 = solve_T2(W, U)
    success = np.all(np.equal(m1, m2))
    if success:
        print "Success"
    else:
        print "Fail"
    return success


def main():
    input_dim = 2
    hidden_dim = 16
    output_dim = 1
    success = 0
    n = 100
    for i in range(n):
        print "Experiment: {}".format(i)
        W = np.random.normal(loc=0, scale=1, size=(input_dim, hidden_dim))
        U = np.random.normal(loc=0, scale=1, size=(hidden_dim, output_dim))
        if experiment(W, U):
            success += 1
    print "Rate: {}/{}".format(success, n)


if __name__ == "__main__":
    main()
