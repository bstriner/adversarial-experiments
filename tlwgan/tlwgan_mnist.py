import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam

from graphs import write_images
from tlwgan_model import TLWGan


def main():
    input_dim = 28 * 28
    latent_dim = 32
    hidden_dim = 256
    epochs = 100
    batches = 256
    batches_d = 32
    batches_l = 32
    opt_d = Adam(1e-3)
    opt_g = Adam(1e-4)
    opt_l = Adam(1e-3)
    batch_size = 64
    mask_count = 8

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    xt = x_train.reshape((x_train.shape[0], input_dim)).astype(np.float32) / 255.0

    def latent_sample(n):
        return np.random.normal(loc=0, scale=1, size=(n, latent_dim)).astype(np.float32)

    def x_sample(n):
        ind = np.random.randint(0, xt.shape[0], (n,))
        return xt[ind, :]

    def gen():
        while True:
            yield x_sample(batch_size), latent_sample(batch_size)

    model = TLWGan(input_dim=input_dim,
                   latent_dim=latent_dim,
                   hidden_dim=hidden_dim,
                   mask_count=mask_count,
                   opt_d=opt_d, opt_g=opt_g, opt_l=opt_l)

    def callback(epoch):
        path = "output/epoch-{:08d}.png".format(epoch)
        z = latent_sample(100)
        xpred = model.generator.predict(z, verbose=0)
        imgs = xpred.reshape((10, 10, 28, 28))
        write_images(path, imgs)

    model.fit(gen=gen(),
              epochs=epochs,
              batches=batches,
              batches_d=batches_d,
              batches_l=batches_l,
              callback=callback)

if __name__ == "__main__":
    main()