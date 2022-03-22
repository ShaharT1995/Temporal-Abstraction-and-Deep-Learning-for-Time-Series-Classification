import time

import numpy as np
from numba import njit, prange
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split

from utils_folder.configuration import ConfigClass
from utils_folder.utils import calculate_metrics

name = "Rocket"


@njit(fastmath=True)
def generate_kernels(input_length, num_kernels, num_channels=1):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels)

    # exponential
    num_channel_indices = (2 ** np.random.uniform(0, np.log2(num_channels + 1), num_kernels)).astype(np.int32)
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros((num_channels, lengths.sum()), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):
        _weights = np.random.normal(0, 1, (num_channels, lengths[i]))

        a = lengths[:i].sum()
        b = a + lengths[i]
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = num_channel_indices[:i].sum()
        b1 = a1 + num_channel_indices[i]
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), num_channel_indices[i], replace=False)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (lengths[i] - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((lengths[i] - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels, stride=1):
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a = 0
        a1 = 0
        for j in range(num_kernels):
            b = a + lengths[j]
            b1 = a1 + num_channel_indices[j]

            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[:, a:b], lengths[j], biases[j], dilations[j], paddings[j],
                             num_channel_indices[j], channel_indices[a1:b1], stride)

            a = b
            a1 = b1

    return _X


class Classifier_Rocket:
    def __init__(self,
                 output_directory: str,
                 n_kernels: int = 10000):
        """
        Initialise the Rocket model

        Inputs:
            output_directory: path to store results/models
            n_kernels: number of random kernels
        """
        config = ConfigClass()
        config.set_seed()

        # super().__init__(output_directory)

        self.name = name
        self.n_kernels = n_kernels
        self.kernels = None
        self.model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)

        self.T = None

        return

    # Taken from Twiesn classifier
    def compute_state_matrix(self, x_in):
        # number of instances
        n = x_in.shape[0]

        # the state matrix to be computed
        x_t = np.zeros((n, self.T, self.N_x), dtype=np.float64)

        # previous state matrix
        x_t_1 = np.zeros((n, self.N_x), dtype=np.float64)

        # loop through each time step
        for t in range(self.T):
            # get all the time series data points for the time step t
            curr_in = x_in[:, t, :]
            # calculate the linear activation
            curr_state = np.tanh(self.W_in.dot(curr_in.T) + self.W.dot(x_t_1.T)).T
            # apply leakage
            curr_state = (1 - self.alpha) * x_t_1 + self.alpha * curr_state
            # save in previous state
            x_t_1 = curr_state
            # save in state matrix
            x_t[:, t, :] = curr_state

        return x_t

    # Taken from Twiesn classifier
    @staticmethod
    def reshape_prediction(y_pred, num_instances, length_series):
        # reshape so the first axis has the number of instances
        new_y_pred = y_pred.reshape(num_instances, length_series, y_pred.shape[-1])
        # average the predictions of instances
        new_y_pred = np.average(new_y_pred, axis=1)
        # get the label with maximum prediction over the last label axis
        new_y_pred = np.argmax(new_y_pred, axis=1)
        return new_y_pred

    def fit(self, x_train, y_train, x_test, y_test, y_true, iter):
        """
        Fit Rocket

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        self.T = x_train.shape[1]

        x_train, x_val, y_train, y_val = \
            train_test_split(x_train, y_train, test_size=0.3, random_state=(42 + iter))

        self.kernels = generate_kernels(x_train.shape[1], self.n_kernels, x_train.shape[2])
        x_training_transform = apply_kernels(x_train, self.kernels)

        start_time = time.perf_counter()
        self.model.fit(x_training_transform, y_train)
        duration = time.time() - start_time

        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

        y_pred = self.model.predict(x_test)
        y_pred = self.reshape_prediction(y_pred, self.x_test.shape[0], self.T)

        df_metrics = calculate_metrics(y_true, y_pred, duration)
        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

