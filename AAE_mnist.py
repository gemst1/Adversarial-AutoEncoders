import tensorflow as tf
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from AAE.model import encoder, decoder, discriminator

class AAE_mnist():
    def __init__(self,
                 n_dim=2,
                 batch_size=100,
                 epochs = 10,
                 log_freq=100,
                 results_path='./results',
                 make_gif=False):

        self.n_dim = n_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_freq = log_freq
        self.results_path=results_path
        self.results_img_path = results_path + "/imges"
        self.make_gif = make_gif

        if not os.path.exists(self.results_img_path):
            os.makedirs(self.results_img_path)
        if self.make_gif and not os.path.exists(self.results_path + "/gif"):
            os.makedirs(self.results_path + "/gif")

        # data load
        self.load_data()
        self.dataset_train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.dtrain_shuffle = self.dataset_train.shuffle(self.x_train.shape[0]).batch(self.batch_size)
        self.dataset_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.dtest_shuffle = self.dataset_test.shuffle(self.x_test.shape[0]).batch(1000)

        # Models
        self.encoder = encoder(n_dim=self.n_dim)
        self.decoder = decoder()
        self.discriminator = discriminator()

        # optimizer
        self.ae_opt = tf.keras.optimizers.Adam(0.0001)
        self.gen_opt = tf.keras.optimizers.Adam(0.0001, beta_1=0, beta_2=0.9)
        self.disc_opt = tf.keras.optimizers.Adam(0.0001, beta_1=0, beta_2=0.9)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        self.x_train = np.reshape(x_train, (-1, 28, 28, 1))
        self.y_train = y_train
        self.x_test = np.reshape(x_test, (-1, 28, 28, 1))
        self.y_test = y_test

    def generator_loss(self, discriminator_on_generator):
        loss = self.loss_object(tf.ones_like(discriminator_on_generator), discriminator_on_generator)
        return loss

    def disc_real_loss(self, discriminator_on_data):
        loss = self.loss_object(tf.ones_like(discriminator_on_data), discriminator_on_data)
        return loss

    def disc_fake_loss(self, discriminator_on_generator):
        loss = self.loss_object(tf.zeros_like(discriminator_on_generator), discriminator_on_generator)
        return loss

    def train(self):
        if self.make_gif:
            self.z_sample = []
            for i in np.linspace(-8, 8, 18):
                for j in np.linspace(-8, 8, 18):
                    self.z_sample.append([i, j])
            self.z_sample = np.asarray(self.z_sample)

        # start training
        for epoch in range(self.epochs):
            print("\nStart of epoch : %d" % (epoch+1))
            for step, (img_batch, label_batch) in enumerate(self.dtrain_shuffle):
                # Autoencoder update
                with tf.GradientTape() as tape:
                    z = self.encoder(img_batch)
                    recon_img = self.decoder(z)

                    recon_loss = tf.reduce_mean(tf.math.squared_difference(img_batch, recon_img))

                grads = tape.gradient(recon_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
                self.ae_opt.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

                # Discriminator update
                z = self.encoder(img_batch)
                z_real = tf.random.normal([self.batch_size, self.n_dim], 0, 8)
                with tf.GradientTape() as tape:
                    real_logits = self.discriminator(z_real)
                    fake_logits = self.discriminator(z)

                    fake_loss = self.disc_fake_loss(fake_logits)
                    real_loss = self.disc_real_loss(real_logits)

                    disc_loss = fake_loss + real_loss

                grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
                self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))

                # Generator update
                with tf.GradientTape() as tape:
                    z = self.encoder(img_batch)
                    gen_logits = self.discriminator(z)
                    gen_loss = self.generator_loss(gen_logits)

                grads = tape.gradient(gen_loss, self.encoder.trainable_weights)
                self.gen_opt.apply_gradients(zip(grads, self.encoder.trainable_weights))

                if (step+1) % self.log_freq == 0:
                    print("epoch %d / %d, step %d / %d" % (epoch+1, self.epochs, step+1, self.x_train.shape[0]//self.batch_size))
                    print("\tgen_loss = %.4f" % (gen_loss))
                    print("\trecon_loss = %.4f" % (recon_loss))
                    print("\tdisc_loss = %.4f" % (disc_loss))
                    print("\t\tdisc_fake_loss = %.4f, disc_real_loss: %.4f" % (fake_loss, real_loss))

                    for test_img_batch, test_label in self.dtest_shuffle.take(1):
                        z = self.encoder(test_img_batch)
                        recon_img = self.decoder(z)

                        plt.scatter(z[:,0], z[:,1], s=0.5, c=test_label)
                        plt.colorbar()
                        plt.title("epoch: %d, step: %d" % (epoch+1, step+1))
                        plt.savefig(self.results_img_path + "/dist_" + str(epoch+1) + "_" + str(step+1))
                        plt.close()
                        # plt.show()

                        for i in range(16):
                            plt.subplot(4, 4, i + 1)
                            plt.imshow((recon_img[i]), cmap='Greys')
                            plt.axis('off')
                        plt.savefig(self.results_img_path + "/dist_" + str(epoch + 1) + "_" + str(step + 1))
                        plt.close()
                        # plt.show()

            if self.make_gif:
                z = self.encoder(self.x_test)
                sample_img = self.decoder(self.z_sample)

                plt.scatter(z[:,0], z[:,1], s=0.5, c=self.y_test)
                plt.colorbar()
                plt.title("epoch: %d" % (epoch + 1))
                plt.xlim((-15, 15))
                plt.ylim((-15, 15))
                plt.savefig(self.results_path + "/gif/dist_" + str(epoch+1))
                plt.close()

                for i in range(18*18):
                    plt.subplot(18, 18, i+1)
                    plt.imshow((sample_img[i]), cmap='Greys')
                    plt.axis('off')
                plt.savefig(self.results_path + "/gif/gen_" + str(epoch+1))
                plt.close()


        # save model
        self.encoder.save(self.results_path + "/encoder")
        self.decoder.save(self.results_path + "/decoder")
        self.discriminator.save(self.results_path + "/discriminator")

        if self.make_gif:
            imgs_array = [np.array(imageio.imread(self.results_path + "/gif/dist_" + str(i+1) + ".png")) for i in range(self.epochs)]
            imageio.mimsave(self.results_path + "/gif/dist.gif", imgs_array, duration=0.1)

            imgs_array = [np.array(imageio.imread(self.results_path + "/gif/gen_" + str(i + 1) + ".png")) for i in range(self.epochs)]
            imageio.mimsave(self.results_path + "/gif/gen.gif", imgs_array, duration=0.1)


if __name__ == "__main__":
    AAE_mnist = AAE_mnist(n_dim=2,
                          epochs=50,
                          log_freq=600,
                          results_path='./results_mnist_AAE',
                          make_gif=True)
    AAE_mnist.train()