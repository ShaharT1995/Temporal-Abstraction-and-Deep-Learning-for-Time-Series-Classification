import math

import tensorflow as tf
import numpy as np
import time
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ELU, Dense, Reshape, concatenate, Permute
from tensorflow.keras.layers import GlobalAveragePooling1D, LSTM, Dropout, Activation, multiply
from tensorflow.python.keras.callbacks import EarlyStopping

from utils_folder.utils import save_logs, calculate_metrics
from utils_folder.configuration import ConfigClass
from sklearn.model_selection import train_test_split

# TODO
# https://github.com/Navidfoumani/Disjoint-CNN

class Classifier_MLSTM_FCN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        config = ConfigClass()
        config.set_seed()

        self.callbacks = None

        if verbose:
            print('Creating MLSTM_FCN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
        self.model.save_weights(self.output_directory + 'model_init.h5')

    def create_class_weight(self, labels, mu=2):
        labels_dict = {}
        total = 0
        for i in range(labels.shape[1]):
            labels_dict.update({i: sum(labels[:, i])})
            total += sum(labels[:, i])

        keys = labels_dict.keys()
        class_weight = dict()

        for key in keys:
            score = math.log(mu * total / float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight

    def build_model(self, input_shape, nb_classes):

        Y_input = Input(shape=input_shape)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(Y_input)
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

        x = Permute((2, 1))(Y_input)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=Y_input, outputs=out)

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        file_path = self.output_directory + 'best_model.h5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        self.callbacks = [es, reduce_lr, model_checkpoint]

        return model

    def fit(self, x, y, x_test, y_test, y_true, iteration):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        batch_size = 16
        nb_epochs = 2000
        mini_batch_size = int(min(x.shape[0] / 10, batch_size))

        # create class weights based on the y label proportions for each image
        class_weight = self.create_class_weight(y)

        start_time = time.time()
        # train the model
        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.3, random_state=(42 + iteration))

        self.hist = self.model.fit(x_train, y_train,
                                   validation_data=[x_val, y_val],
                                   class_weight=class_weight,
                                   verbose=self.verbose,
                                   epochs=nb_epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)

        self.duration = time.time() - start_time

        keras.models.save_model(self.model, self.output_directory + 'model.h5')
        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se