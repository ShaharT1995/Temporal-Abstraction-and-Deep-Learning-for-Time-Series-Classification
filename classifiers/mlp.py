# MLP model 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import matplotlib

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

matplotlib.use('agg')

from utils_folder.utils import save_logs, calculate_metrics
from utils_folder.configuration import ConfigClass


class Classifier_MLP:

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
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        # Reduce learning rate when a metric has stopped improving
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=0.1)

        file_path = self.output_directory + 'best_model.hdf5'

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [es, reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test, y_val, y_true, iteration):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # OLD batch and and epochs
        # batch_size = 16
        nb_epochs = 5000

        # New batch and and epochs
        batch_size = 128
        nb_epochs = nb_epochs // 10

        # Was here before
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

        save_logs(self.output_directory, hist, y_pred, y_true, learning_time, predicting_time,y_pred_prob)

        keras.backend.clear_session()
