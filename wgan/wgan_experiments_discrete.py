
import numpy as np
import keras.backend as K
from keras.objectives import binary_crossentropy
from keras.layers import Activation
from constraints import L0Constraint, L1Constraint
from gan_model_discrete import GanModelDiscrete

def targets_discriminator_gan(n):
    return [np.zeros((n, 1)), np.ones((n, 1))]


def targets_discriminator_wgan(n):
    return [np.ones((n, 1)) * -1, np.ones((n, 1))]


def targets_generator_gan(n):
    return np.ones((n, 1))


def targets_generator_wgan(n):
    return np.ones((n, 1))


def loss_wgan(ytrue, ypred):
    return K.mean(-ytrue * ypred, axis=None)


def loss_gan(ytrue, ypred):
    return binary_crossentropy(ytrue, ypred)


def main():
    nb_epoch = 100
    nb_batch = 256
    k = 5
    batch_size = k

    nb_batch_discriminator = 64
    output_dim = 2
    latent_dim = 2
    W_constraint = lambda: None
    b_constraint = lambda: None
    loss = loss_gan
    activation = lambda: Activation("sigmoid")
    targets_discriminator = targets_discriminator_gan
    targets_generator = targets_generator_gan
    # Typical GAN with nb_batch_discriminator = 5
    # gan = GanModelDiscrete(output_dim, latent_dim, W_constraint, b_constraint, loss, activation, targets_discriminator, targets_generator, k)
    # gan.train(nb_epoch, nb_batch, nb_batch_discriminator, batch_size, "output/gan-d5/epoch-{:06d}.png")

    nb_batch_discriminator = 256
    # Typical GAN with nb_batch_discriminator = 64
    # gan = GanModelDiscrete(output_dim, latent_dim, W_constraint, b_constraint, loss, activation, targets_discriminator, targets_generator, k)
    # gan.train(nb_epoch, nb_batch, nb_batch_discriminator, batch_size, "output/gan-d64/epoch-{:06d}.png")

    W_constraint = lambda: L0Constraint(1e-1)
    b_constraint = lambda: None
    loss = loss_wgan
    activation = lambda: Activation("linear")
    targets_discriminator = targets_discriminator_wgan
    targets_generator = targets_generator_wgan
    # WGAN with l0 clipping = 1e-2
    wganl0 = GanModelDiscrete(output_dim, latent_dim, W_constraint, b_constraint, loss, activation, targets_discriminator,
                      targets_generator, k)
    wganl0.train(nb_epoch, nb_batch, nb_batch_discriminator, batch_size, "output/discrete/wgan-l0/epoch-{:06d}.png")

    W_constraint = lambda: L1Constraint(1.0)
    wganl1 = GanModelDiscrete(output_dim, latent_dim, W_constraint, b_constraint, loss, activation, targets_discriminator, targets_generator, k)
    #wganl1.train(nb_epoch, nb_batch, nb_batch_discriminator, batch_size, "output/discrete/wgan-l1/epoch-{:06d}.png")


if __name__ == "__main__":
    main()
