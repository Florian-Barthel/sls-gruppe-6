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
    ):
        self.c_val = c_val
        self.c_h = c_h

        init = keras.initializers.he_uniform()
        inputs = keras.Input(shape=(16, 16, 1), name="input")
        conv1 = layers.Conv2D(16, 5, strides=(1, 1), activation="relu", kernel_initializer=init)(inputs)
        conv2 = layers.Conv2D(32, 3, strides=(1, 1), activation="relu", kernel_initializer=init)(conv1)
        flatten = layers.Flatten()(conv2)
        dense1 = layers.Dense(128, activation="relu", kernel_initializer=init)(flatten)
        output_policy = layers.Dense(8, activation='softmax', kernel_initializer=init)(dense1)
        output_value = layers.Dense(1, kernel_initializer=init)(dense1)
        self.model = keras.Model(inputs=[inputs], outputs=[output_policy, output_value])

        value_target = K.placeholder(shape=[None, 1])
        action_one_hot = K.placeholder(shape=[None, 8])
        a2c_loss = self.a2c_loss(x=self.model.output, value_target=value_target, action_one_hot=action_one_hot)

        self.optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        update = self.optimizer.get_updates(loss=a2c_loss, params=self.model.trainable_weights)
        self.fit = K.function(inputs=[self.model.input, value_target, action_one_hot], outputs=[a2c_loss], updates=update)

    def predict_both(self, x):
        output_policy, output_value = self.model.predict_on_batch(x)
        return np.argmax(output_policy, axis=-1), output_value

    def save_model(self, filename):
        self.model.save(filename, save_format='h5')

    def load_model(self, filename):
        self.model = keras.models.load_model(filename, compile=False)

    def a2c_loss(self, x, value_target, action_one_hot):
        output_policy, output_value = x
        advantage = tf.stop_gradient(value_target - output_value)
        selected_action = K.sum(action_one_hot * output_policy, axis=-1, keepdims=True)

        policy_loss = -K.mean(advantage * K.log(selected_action))
        value_loss = K.mean(K.square(value_target - output_value))
        h_policy = -K.sum(output_policy * K.log(K.clip(output_policy, 0.00001, 0.99999)), axis=-1, keepdims=True)
        policy_entropy = -K.mean(h_policy)
        return policy_loss + self.c_val * value_loss + self.c_h * policy_entropy


