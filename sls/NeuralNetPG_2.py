import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('TENSORFLOW', tf.version.VERSION)


class Network:
    def __init__(self):
        init = keras.initializers.he_uniform()
        inputs = keras.Input(shape=(2,), name="input")
        x1 = layers.Dense(128, activation="relu", kernel_initializer=init)(inputs)
        x2 = layers.Dense(256, activation="relu", kernel_initializer=init)(x1)
        outputs = layers.Dense(8, activation='softmax')(x2)
        self.model_train = keras.Model(inputs=inputs, outputs=outputs)

        target = keras.backend.placeholder()
        prediction = self.model_train.output
        error = -keras.backend.mean(-keras.backend.log(prediction) * target)

        # self.optimizer = keras.optimizers.RMSprop(lr=0.001)
        self.optimizer = keras.optimizers.RMSprop(learning_rate=0.00025)
        update = self.optimizer.get_updates(loss=error, params=self.model_train.trainable_weights)

        # self.model_train.compile(loss=self.gradient_ascent, optimizer=self.optimizer)
        self.fit = keras.backend.function(inputs=[self.model_train.input, target], outputs=[error], updates=update)

    def train_step_train_model(self, x, y):
        return self.fit(x, y)

    def predict(self, x):
        return self.model_train.predict_on_batch(x)

    def save_model(self, filename):
        self.model_train.save(filename, save_format='h5')

    def load_model(self, filename):
        # manually downgrade h5py to version 2.10.0
        # https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi
        self.model_train = keras.models.load_model(filename, compile=False)


