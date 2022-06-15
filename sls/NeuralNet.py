import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('TENSORFLOW', tf.version.VERSION)


class Network:
    def __init__(self):
        init = keras.initializers.he_uniform()
        inputs = keras.Input(shape=(2,), name="input")
        x1 = layers.Dense(16, activation="relu", kernel_initializer=init)(inputs)
        x2 = layers.Dense(32, activation="relu", kernel_initializer=init)(x1)
        outputs = layers.Dense(8, name="predictions", kernel_initializer=init)(x2)
        self.model_train = keras.Model(inputs=inputs, outputs=outputs)

        # self.optimizer = keras.optimizers.RMSprop(lr=0.001)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001) # best 0.001
        self.model_train.compile(loss=self.mse_clip, optimizer=self.optimizer)
        self.model_static = keras.models.clone_model(self.model_train)

    def train_step(self, x, y):
        return self.model_train.train_on_batch(x, y)

    def predict_static_model(self, x):
        return self.model_static.predict_on_batch(x)

    def predict_train_model(self, x):
        return self.model_train.predict_on_batch(x)

    def update_static_model(self):
        self.model_static.set_weights(self.model_train.get_weights())

    def save_model(self, filename):
        self.model_train.save(filename, save_format='h5')

    def load_model(self, filename):
        # manually downgrade h5py to version 2.10.0 (https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi)
        self.model_train = keras.models.load_model(filename, compile=False)

    @staticmethod
    def mse_clip(x, y):
        return keras.backend.mean(keras.backend.square(keras.backend.clip(x - y, -1, 1)), axis=-1)

