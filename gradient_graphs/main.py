import numpy as np
from datasets import load_datasets
from wgan import wgan_gradients
from gan import gan_gradients
from mbwgan import mbwgan_gradients
from mbwugan import mbwugan_gradients
import os
from graph import graph_gradients, write_gradients, write_weights
from constraints import L1Constraint, LInfConstraint, L1ConstraintHard, L1ConstraintByAxis
from keras.regularizers import L1L2
from regularizers import LtestRegularizer
from tlwgan import tlwgan_gradients
from tlwgan2 import tlwgan2_gradients
def load_models():
    models = []
    nb_batch = 1024
    lr = 3e-4
    hidden_dim = 256
    models.append(('gan',
                   gan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr)))
    models.append(('gan-reg',
                   gan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr, reg=lambda: L1L2(1e-4, 1e-4))))
    models.append(('wgan-linf-constraint',
                   wgan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr, con=lambda: LInfConstraint(1e-2))))
    models.append(('wgan-l1-reg',
                   wgan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr, reg=lambda: L1L2(1e-2, 0))))
    models.append(('wgan-l2-reg',
                   wgan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr, reg=lambda: L1L2(0, 1e-2))))
    models.append(('tlwgan',
                   tlwgan_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr)))
    models.append(('tlwgan-3layer',
                   tlwgan2_gradients(nb_batch=nb_batch, hidden_dim=hidden_dim, lr=lr)))

    """
    nb_batch = 2048
    models.append(('mbwgan-linf-128hidden'.format(nb_batch),
                   mbwgan_gradients(nb_batch=nb_batch, hidden_dim=128, lr=1e-3, con=lambda: LInfConstraint(1e-1))))
    models.append(('mbwgan-linf-128hidden1e2'.format(nb_batch),
                   mbwgan_gradients(nb_batch=nb_batch, hidden_dim=128, lr=1e-3, con=lambda: LInfConstraint(1e-2))))
    models.append(('mbwgan-linf-16hidden'.format(nb_batch),
                   mbwgan_gradients(nb_batch=nb_batch, hidden_dim=16, lr=1e-3, con=lambda: LInfConstraint(1e-1))))
    models.append(('mbwgan-linf-16hidden1e2'.format(nb_batch),
                   mbwgan_gradients(nb_batch=nb_batch, hidden_dim=16, lr=1e-3, con=lambda: LInfConstraint(1e-2))))

    models.append(('mbwgan-reg-128hidden'.format(nb_batch),
                   mbwgan_gradients(nb_batch=nb_batch, hidden_dim=128, lr=1e-3,
                                    con=lambda: None,
                                    reg=lambda: L1L2(1e1, 1e1))))
    
    models.append(("mbwugan-linf-1e1-unroll5", mbwugan_gradients(nb_pretrain_batch=2048, nb_unroll_batch=5, lr=1e-4,
                                                                 con=lambda: LInfConstraint(1e-1), hidden_dim=512)))
    models.append(("mbwugan-linf-1e1-unroll64", mbwugan_gradients(nb_pretrain_batch=2048, nb_unroll_batch=64, lr=1e-4,
                                                                  con=lambda: LInfConstraint(1e-1), hidden_dim=512)))
    models.append(
        ("mbwugan-linf-1e1-unroll64-lr3", mbwugan_gradients(nb_pretrain_batch=2048, nb_unroll_batch=64, lr=1e-3,
                                                            con=lambda: LInfConstraint(1e-1), hidden_dim=512)))
    models.append(("mbwugan-linf-1e2-unroll64", mbwugan_gradients(nb_pretrain_batch=2048, nb_unroll_batch=64, lr=1e-4,
                                                                  con=lambda: LInfConstraint(1e-2), hidden_dim=512)))
    """
    return models


def main():
    dss = load_datasets()
    models = load_models()
    for mname, model in models:
        print "Testing: {}".format(mname)
        for dsname, xreal, xfake in dss:
            path = os.path.join("output", "models", mname, dsname)
            path_png = path + ".png"
            path_txt = path + ".txt"
            if not os.path.exists(path_png):
                xfake_gradients = model(xreal, xfake)
                graph_gradients(path_png, xreal, xfake, xfake_gradients)
                write_gradients(path_txt, xfake, xfake_gradients)
                if isinstance(xfake_gradients, tuple) and len(xfake_gradients) > 2:
                    write_weights(path, xfake_gradients[2])


if __name__ == "__main__":
    main()
