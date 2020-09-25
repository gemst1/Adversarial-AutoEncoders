import tensorflow as tf

class encoder(tf.keras.Model):
    def __init__(self, n_dim=2, name="encoder"):
        super(encoder, self).__init__(name=name)
        self.n_dim = n_dim

        # sub layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(1000)
        self.dense3 = tf.keras.layers.Dense(self.n_dim)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x

class decoder(tf.keras.Model):
    def __init__(self, name="decoder"):
        super(decoder, self).__init__(name=name)

        # sub layers
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(1000)
        self.dense3 = tf.keras.layers.Dense(28*28)
        self.relu = tf.keras.layers.ReLU()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        x = tf.reshape(x, (-1, 28, 28, 1))

        return x

class translator(tf.keras.Model):
    def __init__(self, n_dim=2, name="translator"):
        super(translator, self).__init__(name=name)
        self.n_dim = n_dim

        # sub layers
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(1000)
        self.dense3 = tf.keras.layers.Dense(self.n_dim)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x

class discriminator(tf.keras.Model):
    def __init__(self, name="discriminator"):
        super(discriminator, self).__init__(name=name)

        # sub layers
        self.dense1 = tf.keras.layers.Dense(1000)
        self.dense2 = tf.keras.layers.Dense(1000)
        self.dense3 = tf.keras.layers.Dense(1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x