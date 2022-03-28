# FCN - LSTM model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
import tensorflow as tf
import numpy as np
import time

from tensorflow.python.keras.callbacks import EarlyStopping

from utils_folder.configuration import ConfigClass
from utils_folder.utils import save_logs, calculate_metrics

# TODO
# https://github.com/lilly9117/Sensor-data-classification/blob/77166bec9339965b1bfff58cb477853ef92e7f0d/time_series_classification/classifiers/lstm_fcn.py


def squeeze_excite_block(input):
    """
    Create a squeeze-excite block
    Args:
        input: input tensor
    Returns: a keras tensor
    """
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


class Classifier_LSTMFCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build:
            config = ConfigClass()
            config.set_seed()

            self.output_directory = output_directory
            self.callbacks = None

            self.model = self.build_model(input_shape, nb_classes)

            # if verbose:
            #     self.model.summary()

            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        ip = Input(input_shape)

        x = Masking()(ip)
        x = LSTM(64)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)

        model = Model(ip, out)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [es, reduce_lr, model_checkpoint]

        return model

    def fit(self, x, y, x_test, y_test, y_true, iteration):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # OLD batch and and epochs
        # batch_size = 8
        nb_epochs = 100

        # New batch and and epochs
        batch_size = 128
        nb_epochs = nb_epochs // 10

        # Was here before
        mini_batch_size = int(min(x.shape[0] / 10, batch_size))

        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.3, random_state=(42 + iteration))

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

        save_logs(self.output_directory, hist, y_pred, y_true, learning_time, predicting_time)

        keras.backend.clear_session()
