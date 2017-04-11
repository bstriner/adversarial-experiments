import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import matplotlib.pyplot as plt
import os
import theano
import theano.tensor as T
import numpy as np
from keras.initializations import glorot_uniform, zero
import keras.backend as K
import itertools
from tqdm import tqdm


def layer_norm(x):
    m = T.mean(x, axis=-1)
    std = T.sqrt(T.var(x, axis=-1))
    return (x-m)/(std + 1e-6)

class GanModelDiscreteMinibatch(object):
    def __init__(self, k):
        self.model = None
        self.x_fake = None
        self.x_real = None
        self.train_g_function = None
        self.train_d_function = None
        self.k = k
        self.create_model()

    def x_real_sample(self):
        z = np.arange(self.k) * 2 * np.pi / self.k
        x = np.concatenate((np.cos(z).reshape((-1, 1)), np.sin(z).reshape((-1, 1))), axis=1)
        return x.astype(np.float32)

    def create_model(self):
        x_fake = np.random.normal(0, 1, (self.k, 2))
        x_fake = theano.shared(x_fake.astype(np.float32), "x_fake")
        self.x_fake = x_fake
        d_hidden = 128
        d_h1_W = glorot_uniform((2 * 2, d_hidden), "d_h1_W")
        d_h1_b = zero((d_hidden,), "d_h1_b")
        d_h2_W = glorot_uniform((d_hidden, d_hidden), "d_h2_W")
        d_h2_b = zero((d_hidden,), "d_h2_b")
        d_h3_W = glorot_uniform((d_hidden, d_hidden), "d_h3_W")
        d_h3_b = zero((d_hidden,), "d_h3_b")
        d_y_W = glorot_uniform((d_hidden, 1), "d_y_W")
        d_y_b = zero((1,), "d_y_b")

        def d(x):
            y = 0.0
            for combo in itertools.permutations(range(self.k), 2):
                x1 = x[combo, :]
                x1 = T.flatten(x1)
                h1 = T.dot(x1, d_h1_W) + d_h1_b
                h1 = layer_norm(h1)
                h1 = T.nnet.relu(h1, 0.2)
                h2 = T.dot(h1, d_h2_W) + d_h2_b
                h2 = layer_norm(h2)
                h2 = T.nnet.relu(h2, 0.2)
                h3 = T.dot(h2, d_h3_W) + d_h3_b
                h3 = layer_norm(h3)
                h3 = T.nnet.relu(h3, 0.2)
                _y = T.dot(h3, d_y_W) + d_y_b
                y += _y
            return y

        self.x_real = theano.shared(self.x_real_sample(), "x_real")
        y_real = d(self.x_real)
        y_fake = d(x_fake)
        g_loss = T.mean(y_real, axis=None) - T.mean(y_fake, axis=None)
        d_loss = T.mean(y_fake, axis=None) - T.mean(y_real, axis=None)

        lr_g = 1e-5
        params_g = [x_fake]
        params_g_t1 = [p - lr_g * T.grad(g_loss, p) for p in params_g]
        updates_g = list(zip(params_g, params_g_t1))
        self.train_g_function = theano.function([], [g_loss, d_loss], updates=updates_g)

        lr_d = 1e-4
        params_d_W = [d_h1_W, d_h2_W, d_h3_W, d_y_W]
        params_d_b = [d_h1_b, d_h2_b, d_h3_b, d_y_b]
        params_d_W_t1 = [p - lr_d * T.grad(d_loss, p) for p in params_d_W]
        params_d_b_t1 = [p - lr_d * T.grad(d_loss, p) for p in params_d_b]
        limit = 1e-1
        params_d_W_t1 = [T.clip(p, -limit, limit) for p in params_d_W_t1]
        updates_d = list(zip(params_d_W + params_d_b, params_d_W_t1 + params_d_b_t1))

        self.train_d_function = theano.function([], [g_loss, d_loss], updates=updates_d)
        self.test_function = theano.function([], [g_loss, d_loss])

    def train(self, nb_epoch, nb_batch, nb_batch_discriminator, path_format):
        for epoch in tqdm(range(nb_epoch), desc="Training"):
            self.write_image(path_format.format(epoch))
            for batch in range(nb_batch):
                # for _ in tqdm(range(nb_batch_discriminator), desc="Training Discriminator"):
                for _ in range(nb_batch_discriminator):
                    self.train_d_function()
                loss = self.test_function()
                count = 0
                while loss[1] > 0:
                    count += 1
                    if count > 512:
                        print("Timed out training discriminator")
                        self.write_image(path_format.format(epoch + 1))
                        return
                    loss = self.train_d_function()
                if batch == 0:
                    loss = self.test_function()
                    tqdm.write("G Loss: {}, D Loss: {}".format(loss[0], loss[1]))
                self.train_g_function()

    def write_image(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        fig = plt.figure()
        x_fake = self.x_fake.get_value()
        x_real = self.x_real.get_value()
        plt.scatter(x_real[:, 0], x_real[:, 1], c="g", s=200)
        plt.scatter(x_fake[:, 0], x_fake[:, 1], c="r", s=200)
        plt.ylim(-2, 2)
        plt.xlim(-2, 2)
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    model = GanModelDiscreteMinibatch(3)
    model.train(200, 64, 32, "output/gan-model-discrete-minibatch-3/epoch-{:06d}.png")
    model = GanModelDiscreteMinibatch(5)
    model.train(200, 64, 32, "output/gan-model-discrete-minibatch-5/epoch-{:06d}.png")
