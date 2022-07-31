import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np

print('TENSORFLOW', tf.version.VERSION)
print(tf.config.experimental_list_devices())


class Network:
    def __init__(
            self,
            c_val=0.5,
            c_h=0.005,
            lr=0.0007,
            n_step=5,
            gamma=0.99
    ):
        self.c_val = c_val
        self.c_h = c_h
        self.n_step = n_step
        self.gamma = gamma

        init = keras.initializers.he_uniform()
        inputs = keras.Input(shape=(16, 16, 1), name="input")
        conv1 = layers.Conv2D(16, 5, strides=(1, 1), activation="relu", kernel_initializer=init)(inputs)
        conv2 = layers.Conv2D(32, 3, strides=(1, 1), activation="relu", kernel_initializer=init)(conv1)
        flatten = layers.Flatten()(conv2)
        dense1 = layers.Dense(128, activation="relu", kernel_initializer=init)(flatten)
        output_policy = layers.Dense(8, activation='softmax', kernel_initializer=init)(dense1)
        output_value = layers.Dense(1, kernel_initializer=init)(dense1)
        self.model = keras.Model(inputs=[inputs], outputs=[output_policy, output_value])

        value_target = K.placeholder(shape=[None, 8])

        a2c_loss = self.a2c_loss(x=self.model.output, value_target=value_target)

        self.optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        # self.model.compile(loss=a2c_loss, optimizer=self.optimizer)
        update = self.optimizer.get_updates(loss=a2c_loss, params=self.model.trainable_weights)
        self.fit = K.function(inputs=[self.model.input, value_target], outputs=[a2c_loss], updates=update)

    def predict_both(self, x):
        output_policy, output_value = self.model.predict_on_batch(x)
        return np.argmax(output_policy, axis=-1), output_value

    def predict_policy(self, x):
        prediction = self.model.predict_on_batch(x)
        return np.argmax(prediction[: 0], axis=-1)

    def save_model(self, filename):
        self.model.save(filename, save_format='h5')

    def load_model(self, filename):
        self.model = keras.models.load_model(filename, compile=False)

    def a2c_loss(self, x, value_target):

        output_policy, output_value = x
        advantage = tf.stop_gradient(value_target - output_value)

        policy_loss = -K.mean(advantage * K.log(output_policy))
        value_loss = K.mean(K.square(value_target - output_value))
        policy_entropy = -K.mean(-K.sum(output_policy * K.log(K.maximum(output_policy, 0.00001)), axis=-1))
        return policy_loss + self.c_val * value_loss + self.c_h * policy_entropy


