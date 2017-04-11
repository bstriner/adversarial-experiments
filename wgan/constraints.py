import numpy as np
from keras.constraints import Constraint
import keras.backend as K


class L0Constraint(Constraint):
    def __init__(self, max_value):
        self.max_value = np.float32(max_value)

    def __call__(self, p):
        return K.clip(p, -self.max_value, self.max_value)


class L1Constraint(Constraint):
    def __init__(self, max_value):
        self.max_value = np.float32(max_value)

    def __call__(self, p):
        l1 = K.sum(K.abs(p), axis=None)
        scale = K.clip(self.max_value/l1, 0, 1)
        return p * scale

class L1ConstraintHard(Constraint):
    def __init__(self, max_value):
        self.max_value = np.float32(max_value)

    def __call__(self, p):
        l1 = K.sum(K.abs(p), axis=None)
        return p / l1
