import time
import numpy as np
import pickle
import os
from numba import njit, prange

from sklearn.linear_model import RidgeClassifierCV
from utils_folder.utils import calculate_metrics


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64,Tuple((int32,int32,int32)))")
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


@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel=True,
      fastmath=True)
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


class RocketClassifier:
    """
    This is a class implementing Rocket for time series classifier.
    The code is adapted by the authors from the original Rocket implementation at https://github.com/angus924/rocket
    """
    # was n_kernels: int = 10000
    def __init__(self,
                 output_folder: str,
                 build: bool = True,
                 n_kernels: int = 1000):
        self.name = "Rocket"
        self.output_folder = output_folder
        self.n_kernels = n_kernels
        self.kernels = None
        self.classifier = None
        if build:
            self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=10, normalize=True)

        self.model_path = os.path.join(self.output_folder, 'model.pkl')

    def get_fitted_model(self):
        val_to_ret = None
        # load it
        with open(self.model_path, 'rb') as file:
            val_to_ret = pickle.load(file)
        return val_to_ret

    def fit(self, x_train, y_train, x_test, y_val, y_true, iteration):
        """
        Fit Rocket

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        self.kernels = generate_kernels(x_train.shape[1], self.n_kernels, x_train.shape[2])
        x_training_transform = apply_kernels(x_train, self.kernels)

        y_train_new = np.argmax(y_train, axis=1)

        start_time = time.time()
        self.classifier.fit(x_training_transform, y_train_new)
        learning_time = time.time() - start_time

        start_time = time.time()
        y_pred = self.predict(x_test)
        predicting_time = time.time() - start_time

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        # Save Metrics
        df_metrics = calculate_metrics(y_true, y_pred, learning_time, predicting_time)
        df_metrics.to_csv(self.output_dir + 'df_metrics.csv', index=False)

    def predict(self, x_test: np.array):
        """
        Do prediction with Rocket

        Inputs:
            x_test: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        start_time = time.perf_counter()
        x_test_transform = apply_kernels(x_test, self.kernels)

        y_pred_des_func = self.classifier.decision_function(x_test_transform)
        y_pred = []
        for v in y_pred_des_func:
            y_pred.append(np.exp(v) / (1 + np.exp(v)))

        test_duration = time.perf_counter() - start_time

        return y_pred
