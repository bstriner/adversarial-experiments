from keras.regularizers import Regularizer
import numpy as np
import itertools
import theano.tensor as T


class LtestRegularizer(Regularizer):
    """Regularizer base class.
    """

    def __init__(self, weight):
        self.weight = np.float32(weight)

    def __call__(self, x):
        shape = x._keras_shape
        print "Shape: {}".format(shape)
        # raise ValueError("Shape:{}".format(shape))
        assert (shape[1] < 10)
        l1s = []
        for combo in itertools.combinations_with_replacement([-1, 1], shape[1]):
            tot = T.zeros_like(x[:, 0])
            for i, c in enumerate(combo):
                tot += c * x[:, i]
            l1 = T.reshape(T.sum(T.abs_(tot)), (1,))
            l1s.append(l1)
        l1s = T.concatenate(l1s, axis=0)
        maxl1 = T.max(l1s, axis=0)
        return maxl1 * self.weight
