# Our proposed model CNN + LSTM
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from utils_folder.utils import save_logs, calculate_metrics
from utils_folder.configuration import ConfigClass


class Classifier_ENCODER:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        config = ConfigClass()
        config.set_seed()

        self.output_directory = output_directory
        self.callbacks = None

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)

        # conv1 = tfa.layers.InstanceNormalization()(conv1) - The old line
        conv1 = tf.keras.layers.BatchNormalization(axis=1)(conv1)

        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)

        # conv2 = tfa.layers.InstanceNormalization()(conv2) - The old line
        conv2 = tf.keras.layers.BatchNormalization(axis=1)(conv2)

        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)

        # conv3 = tfa.layers.InstanceNormalization()(conv3) - The old line
        conv3 = tf.keras.layers.BatchNormalization(axis=1)(conv3)

        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)

        # dense_layer = tfa.layers.InstanceNormalization()(dense_layer) - The old line
        dense_layer = tf.keras.layers.BatchNormalization(axis=1)(dense_layer)

        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_loss', save_best_only=True)

        self.callbacks = [es, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test, y_test, y_true, iter):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # OLD batch and and epochs
        # batch_size = 12
        nb_epochs = 100

        # New batch and and epochs
        batch_size = 128
        nb_epochs = nb_epochs // 10
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))


        # Added lines because model's fit on the testing set - bug in the original code
        x_train, x_val, y_train, y_val = \
            train_test_split(x_train, y_train, test_size=0.3, random_state=(42 + iter))

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        learning_time = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        start_time = time.time()
        y_pred = model.predict(x_test)
        predicting_time = time.time() - start_time

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, learning_time, predicting_time, lr=False)

        keras.backend.clear_session()
