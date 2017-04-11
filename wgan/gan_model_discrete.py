import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

from gan_model import GanModel
import numpy as np
from keras.layers import Layer, Input, Dense
from keras.initializations import uniform
from keras.models import Model
from keras.engine import InputSpec
import keras.backend as K
from keras.optimizers import RMSprop


def generator_init(shape, name):
    return uniform(shape, name=name, scale=1)

class GeneratorDiscrete(Layer):
    def __init__(self, k):
        self.k = k
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_dim = 2
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim=2)]
        self.x_fake = self.add_weight((self.k, 2),
                                      initializer=generator_init,
                                      name='{}_x_fake'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return x*self.x_fake
        #return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class GanModelDiscrete(GanModel):
    def latent_sample(self, n):
        # x=np.arange(self.k).reshape((-1,1))
        # m = np.mean(x, keepdims=True, axis=None)
        # std = np.std(x, keepdims=True, axis=None)
        # return (x-m)/(std)
        return np.ones((self.k, 2)).astype(np.float32)

    def real_sample(self, n):
        z = np.arange(self.k) * 2 * np.pi / self.k
        x = np.concatenate((np.cos(z).reshape((-1, 1)), np.sin(z).reshape((-1, 1))), axis=1)
        return x

    def create_generator(self):
        # z = Input(batch_shape=(self.k, self.latent_dim,))
        z = Input(batch_shape=(self.k, self.latent_dim))
        x = GeneratorDiscrete(self.k)(z)
        self.generator = Model([z], [x])
        print("Generator")
        self.generator.summary()

    def create_discriminator(self):
        hidden_dim = 128
        x = Input(batch_shape=(self.k,self.output_dim))
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(x)
        # h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        # h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(hidden_dim, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        # h = LayerNormalization()(h)
        h = self.hidden_activation()(h)
        h = Dense(1, W_constraint=self.W_constraint(), b_constraint=self.b_constraint())(h)
        y = self.activation()(h)
        self.discriminator = Model([x], [y])
        print("Discriminator")
        self.discriminator.summary()

    def create_model(self):
        self.create_discriminator()
        self.create_generator()
        x_real = Input(batch_shape=(self.k, self.output_dim), name="x_real")
        z = Input(batch_shape=(self.k, self.latent_dim), name="z")
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
