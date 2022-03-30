import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import time

from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

from utils_folder.utils import save_logs, calculate_metrics
from utils_folder.configuration import ConfigClass


class Classifier_MCDCNN:
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
        return

    # TODO Shahar - I dont know what we did here, the two function is identical, keep for backup

    # def build_NetFlowAndWafer_model(self, input_shape, nb_classes):
    #     n_t = input_shape[0]
    #     n_vars = input_shape[1]
    #
    #     padding = 'valid'
    #     # for Italy Power On demand
    #     if n_t < 60:
    #         padding = 'same'
    #
    #     input_layers = []
    #     conv2_layers = []
    #
    #     for n_var in range(n_vars):
    #         input_layer = keras.layers.Input((n_t, 1))
    #         input_layers.append(input_layer)
    #
    #         conv1_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(input_layer)
    #         conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)
    #
    #         conv2_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(conv1_layer)
    #         conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
    #         conv2_layer = keras.layers.Flatten()(conv2_layer)
    #
    #         conv2_layers.append(conv2_layer)
    #
    #     if n_vars == 1:
    #         # to work with univariate time series
    #         concat_layer = conv2_layers[0]
    #     else:
    #         concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)
    #
    #     # init = tf.keras.initializers.GlorotNormal(seed=123)
    #     # kernel_initializer = init
    #     fully_connected = keras.layers.Dense(units=732, activation='tanh')(concat_layer)
    #
    #     output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)
    #
    #     model = keras.models.Model(inputs=input_layers, outputs=output_layer)
    #
    #     model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
    #                   optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005))
    #
    #     file_path = self.output_directory + 'best_model.hdf5'
    #
    #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
    #     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
    #                                                        save_best_only=True)
    #
    #     self.callbacks = [es, model_checkpoint]
    #
    #     return model

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        # No padding
        padding = 'valid'
        if n_t < 60:
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732, activation='tanh')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                                                                      decay=0.0005),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [es, model_checkpoint]

        return model

    def prepare_input(self, x):
        n_vars = x.shape[2]

        new_x = []

        for i in range(n_vars):
            new_x.append(x[:, :, i: i+1])

        return new_x

    def fit(self, x, y, x_test, y_test, y_true, iteration):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # OLD batch and and epochs
        # mini_batch_size = 16
        nb_epochs = 120

        # New batch and and epochs
        batch_size = 128
        nb_epochs = nb_epochs // 10
        mini_batch_size = int(min(x.shape[0] / 10, batch_size))

        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.3, random_state=(42 + iteration))

        x_test = self.prepare_input(x_test)
        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()
