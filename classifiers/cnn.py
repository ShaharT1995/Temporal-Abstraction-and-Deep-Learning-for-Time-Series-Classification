# CNN model
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from utils_folder.utils import save_logs, calculate_metrics
from utils_folder.configuration import ConfigClass


class Classifier_CNN:
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

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        # for italy power on demand dataset
        if input_shape[0] < 60:
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [es, model_checkpoint]
        return model

    def fit(self, x_train, y_train, x_test, y_test, y_true,  iteration):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # OLD batch and and epochs
        # mini_batch_size = 16
        nb_epochs = 2000

        # New batch and and epochs
        batch_size = 128
        nb_epochs = nb_epochs // 10
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        # Added lines because model's fit on the testing set - bug in the original code
        x_train, x_val, y_train, y_val = \
            train_test_split(x_train, y_train, test_size=0.3, random_state=(42 + iteration))

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        learning_time = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        start_time = time.time()
        y_pred = model.predict(x_test)
        predicting_time = time.time() - start_time
        y_pred_prob = y_pred

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, learning_time, predicting_time,y_pred_prob, lr=False)

        keras.backend.clear_session()