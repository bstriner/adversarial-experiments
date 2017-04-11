from keras.layers import Dense, Input, Activation, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import os
import matplotlib.pyplot as plt
import numpy as np
from gmm_dataset import gmm_dataset
from tqdm import tqdm
from normalization import LayerNormalization
from keras.layers.normalization import BatchNormalization


class GanModel(object):
    def __init__(self, output_dim, latent_dim, W_constraint, b_constraint, loss, activation, targets_discriminator,
                 targets_generator,
                 k):
        self.model = None
        self.generator = None
        self.discriminator = None
        self.gan_generator = None
        self.gan_discriminator = None
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint
        self.loss = loss
        self.activation = activation
        self.targets_discriminator = targets_discriminator
        self.targets_generator = targets_generator
        self.k = k
        self.hidden_activation = lambda: LeakyReLU(0.2)
        # self.hidden_activation = lambda: Activation("relu")
        self.create_model()

    def create_discriminator(self):
        hidden_dim = 128
        x = Input((self.output_dim,))
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(x)
        #h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        #h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        #h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(1, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        y = self.activation()(h)
        self.discriminator = Model([x], [y])
        print("Discriminator")
        self.discriminator.summary()

    def create_generator(self):
        hidden_dim = 512
        z = Input((self.latent_dim,))
        h = Dense(hidden_dim)(z)
        h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim)(h)
        h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim)(h)
        h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        x = Dense(self.output_dim)(h)
        self.generator = Model([z], [x])
        print("Generator")
        self.generator.summary()

    def latent_sample(self, n):
        return np.random.normal(0, 1, (n, self.latent_dim))

    def real_sample(self, n):
        return gmm_dataset(self.k, n)

    def trainable_generator(self, trainable_generator):
        self.generator.trainable = trainable_generator
        for layer in self.generator.layers:
            layer.trainable = trainable_generator
        self.discriminator.trainable = not trainable_generator
        for layer in self.discriminator.layers:
            layer.trainable = not trainable_generator

    def create_model(self):
        self.create_discriminator()
        self.create_generator()
        x_real = Input((self.output_dim,), "x_real")
        z = Input((self.latent_dim,), "z")
        print("X: {}".format(x_real.ndim))
        print("Z: {}".format(z.ndim))
        x_fake = self.generator(z)
        y_fake = self.discriminator(x_fake)
        y_real = self.discriminator(x_real)
        self.gan_generator = Model([z], [y_fake])
        self.trainable_generator(True)
        opt_g = RMSprop(1e-4)
        self.gan_generator.compile(opt_g, loss=self.loss)
        self.gan_discriminator = Model([x_real, z], [y_fake, y_real])
        self.trainable_generator(False)
        opt_d = RMSprop(1e-4)
        self.gan_discriminator.compile(opt_d, loss=self.loss)

    def train(self, nb_epoch, nb_batch, nb_batch_discriminator, batch_size, path):
        for epoch in tqdm(range(nb_epoch), desc="Training"):
            d_losses = []
            g_losses = []
            for batch in tqdm(range(nb_batch), desc="Epoch {}".format(epoch)):
                self.trainable_generator(False)
                for _ in range(nb_batch_discriminator):
                    z = self.latent_sample(batch_size)
                    x = self.real_sample(batch_size)
                    discriminator_targets = self.targets_discriminator(batch_size)
                    d_loss = self.gan_discriminator.train_on_batch([x, z], discriminator_targets)[0]
                    d_losses.append(d_loss)
                if batch == 0:
                    imagepath = path.format(epoch)
                    self.write_image(imagepath)
                self.trainable_generator(True)
                z = self.latent_sample(batch_size)
                generator_targets = self.targets_generator(batch_size)
                g_loss = self.gan_generator.train_on_batch([z], generator_targets)
                g_losses.append(g_loss)
            d_loss = np.mean(d_losses, axis=None)
            g_loss = np.mean(g_losses, axis=None)
            print("Epoch: {}, Generator loss: {}, Discriminator loss: {}".format(epoch, g_loss, d_loss))

    def write_image(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        lim = 2
        n = 16 * self.k
        k = 50
        grid = np.meshgrid(np.linspace(-lim, lim, k), np.linspace(lim, -lim, k))
        grid = np.concatenate((np.expand_dims(grid[0], 2), np.expand_dims(grid[1], 2)), axis=2)
        x = grid.reshape((-1, 2))
        y = self.discriminator.predict([x])
        # print("Shapes: {}, {}, {}".format(x.shape, y.shape, grid.shape))
        ygrid = y.reshape(grid.shape[0:2])
        z = self.latent_sample(n)
        x_real = self.real_sample(n)
        x_fake = self.generator.predict([z])

        fig = plt.figure()
        plt.imshow(ygrid, cmap="coolwarm", extent=(-lim, lim, -lim, lim))
        plt.scatter(x_real[:, 0], x_real[:, 1], c="w", s=80)
        plt.scatter(x_fake[:, 0], x_fake[:, 1], c="k", s=80)
        plt.xlim((-lim, lim))
        plt.ylim((-lim, lim))
        fig.savefig(path)
        plt.close(fig)
