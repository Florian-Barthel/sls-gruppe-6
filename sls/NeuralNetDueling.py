import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

print('TENSORFLOW', tf.version.VERSION)


class Network:
    def __init__(self):
        init = keras.initializers.he_uniform()
        inputs = keras.Input(shape=(2,), name="input")
        x1 = layers.Dense(16, activation="relu", kernel_initializer=init)(inputs)
        x2 = layers.Dense(32, activation="relu", kernel_initializer=init)(x1)
        x3 = layers.Dense(9, kernel_initializer=init)(x2)
        merge_va = layers.Lambda(lambda r: r[:, 0:1] + r[:, 1:] - K.mean(r[:, 1:], axis=-1, keepdims=True), output_shape=(8,))(x3)
        self.model_train = keras.Model(inputs=inputs, outputs=merge_va)

        # self.optimizer = keras.optimizers.RMSprop(lr=0.001)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001) # best 0.001
        self.model_train.compile(loss=self.mse_clip, optimizer=self.optimizer)

        self.model_target = keras.models.clone_model(self.model_train)
        self.model_target.compile(loss=self.mse_clip, optimizer=self.optimizer)

    def train_step_train_model(self, x, y):
        return self.model_train.train_on_batch(x, y)

    def train_step_target_model(self, x, y):
        return self.model_target.train_on_batch(x, y)

    def predict_target_model(self, x):
        return self.model_target.predict_on_batch(x)

    def predict_train_model(self, x):
        return self.model_train.predict_on_batch(x)

    def update_target_model(self):
        self.model_target.set_weights(self.model_train.get_weights())

    def save_model(self, filename):
        self.model_train.save(filename, save_format='h5')

    def load_model(self, filename):
        # manually downgrade h5py to version 2.10.0 (https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi)
        self.model_train = keras.models.load_model(filename, compile=False)

    @staticmethod
    def mse_clip(x, y):
        return keras.backend.mean(keras.backend.square(keras.backend.clip(x - y, -1, 1)), axis=-1)

