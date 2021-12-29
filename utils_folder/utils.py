import pickle
import re
from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import operator
import seaborn as sns

from utils_folder.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils_folder.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils_folder.constants import dataset_names_for_archive
from utils_folder.constants import CLASSIFIERS
from utils_folder.constants import ITERATIONS
from utils_folder.constants import MTS_DATASET_NAMES
from utils_folder.configuration import ConfigClass
from utils_folder.ranking_graph import draw_cd_diagram


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder


from scipy.interpolate import interp1d
from scipy.io import loadmat

import random as rn

matplotlib.use('agg')

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

np.random.seed(37)
rn.seed(1254)


def open_pickle(name):
    config = ConfigClass()
    file = open(config.get_path() + "/SyncProject/temporal_abstraction_f/pickle_files//" + name + ".pkl", "rb")
    data = pickle.load(file)
    return data


def write_pickle(name, data):
    config = ConfigClass()
    file = open(config.get_path() + "/SyncProject/temporal_abstraction_f/pickle_files//" + name + ".pkl", "wb")
    pickle.dump(data, file)
    file.close()


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(root_dir, archive_name, dataset_name):
    print("ERROR!!")
    return None
    # datasets_dict = {}
    # cur_root_dir = root_dir.replace('-temp', '')
    #
    # if archive_name == 'mts_archive':
    #     file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    #     x_train = np.load(file_name + 'x_train.npy')
    #     y_train = np.load(file_name + 'y_train.npy')
    #     x_test = np.load(file_name + 'x_test.npy')
    #     y_test = np.load(file_name + 'y_test.npy')
    #
    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())
    #
    # elif archive_name == 'UCRArchive_2018':
    #     root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    #     df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    #
    #     df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)
    #
    #     y_train = df_train.values[:, 0]
    #     y_test = df_test.values[:, 0]
    #
    #     x_train = df_train.drop(columns=[0])
    #     x_test = df_test.drop(columns=[0])
    #
    #     x_train.columns = range(x_train.shape[1])
    #     x_test.columns = range(x_test.shape[1])
    #
    #     x_train = x_train.values
    #     x_test = x_test.values
    #
    #     # znorm
    #     std_ = x_train.std(axis=1, keepdims=True)
    #     std_[std_ == 0] = 1.0
    #     x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    #
    #     std_ = x_test.std(axis=1, keepdims=True)
    #     std_[std_ == 0] = 1.0
    #     x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    #
    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())
    # else:
    #     file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
    #     x_train, y_train = readucr(file_name + '_TRAIN')
    #     x_test, y_test = readucr(file_name + '_TEST')
    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())
    #
    # return datasets_dict


def read_all_datasets(root_dir, archive_name, transformation_name=-1, after_ta=False, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []
    config = ConfigClass()

    # todo
    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            mts_path = config.get_mts_path()
            # root_dir_dataset = cur_root_dir + 'mtsdata/archives/' + archive_name + '/' + dataset_name + '/'
            root_dir_dataset = mts_path + '/' + dataset_name + '/'

            if after_ta:
                path = config.get_prop_path() + ARCHIVE_NAMES[0] + "//" + CLASSIFIERS[0] + "//" + \
                        config.get_method()[0] + "//" + dataset_name + "//"

                x_train = np.load(path + 'transformation2_type' + transformation_name + '_train.npy')
                y_train = np.load(root_dir_dataset + 'y_train.npy')
                x_test = np.load(path + 'transformation2_type' + transformation_name + '_test.npy')
                y_test = np.load(root_dir_dataset + 'y_test.npy')

            else:
                x_train = np.load(root_dir_dataset + 'x_train.npy')
                y_train = np.load(root_dir_dataset + 'y_train.npy')
                x_test = np.load(root_dir_dataset + 'x_test.npy')
                y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive_2018':
        for dataset_name in DATASET_NAMES_2018:
            if after_ta:
                path = config.get_prop_path() + ARCHIVE_NAMES[0] + "//" + CLASSIFIERS[0] + "//" + \
                       config.get_method()[0] + "//" + dataset_name + "//"

                df_train = pd.read_csv(path + '//transformation2_type' + transformation_name + '_Train.csv',
                                       sep=',', header=None)

                df_test = pd.read_csv(path + '//transformation2_type' + transformation_name + '_Test.csv',
                                       sep=',', header=None)

            else:
                ucr_path = config.get_ucr_path()
                root_dir_dataset = ucr_path + '/' + dataset_name + '/'

                df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

                df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

                # Padding with zero for missing values
                df_train = df_train.fillna(0)
                df_test = df_test.fillna(0)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    # else:
    #     for dataset_name in DATASET_NAMES:
    #         root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    #         file_name = root_dir_dataset + dataset_name
    #         x_train, y_train = readucr(file_name + '_TRAIN')
    #         x_test, y_test = readucr(file_name + '_TEST')
    #
    #         datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                        y_test.copy())
    #
    #         dataset_names_to_sort.append((dataset_name, len(x_train)))
    #
    #     dataset_names_to_sort.sort(key=operator.itemgetter(1))
    #
    #     for i in range(len(DATASET_NAMES)):
    #         DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)

        # Make all time series in the same length (by converting them to the length of the maximum series length)
        # by linear interpolation
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x


def transform_mts_to_ucr_format():
    config = ConfigClass()
    mts_root_dir = config.get_mts_path()
    mts_out_dir = config.get_mts_path()

    MTS_DICT = {}
    length_dict = {}

    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir + dataset_name + '/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir + dataset_name + '/' + dataset_name + '.mat')
        # a = loadmat(mts_root_dir  '/' + dataset_name + '.mat')
        a = a['mts']
        a = a[0, 0]

        dt = a.dtype.names
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        MTS_DICT[dataset_name] = {"time_serious_length": max_length, "number_of_entities_train": len(x_train),
                                  "number_of_entities_test": len(x_test), "number_of_attributes": n_var}

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()
        # continue

        x_train = transform_to_same_length(x_train, n_var, max_length)
        x_test = transform_to_same_length(x_test, n_var, max_length)

        length_dict[dataset_name] = max_length

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')

    write_pickle("MTS_Dictionary", MTS_DICT)
    write_pickle("length_dict", length_dict)


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def generate_results_csv(output_file_name, root_dir, classifier, params="", after_ta=False):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    config = ConfigClass()

    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            # todo
            # datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(ITERATIONS):
                if it != 0:
                    curr_archive_name = 'itr' + str(it) + ", " + params + '/'
                else:
                    curr_archive_name = ", " + params + '/'

                for dataset_name in dataset_names_for_archive[archive_name]:
                    output_dir = root_dir + '/' + archive_name + '/results/' + classifier_name + '/' + \
                                 config.get_method()[0] + "//" + curr_archive_name + '/' + dataset_name + '/' \
                                 + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    df_metrics['iteration'] = it
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    # todo
    path = root_dir + "Results//"
    if after_ta:
        path += "ResultsAfterTA"

    if not os.path.exists(path + "//" + classifier + "//"):
        os.makedirs(path + "//" + classifier + "//")

    res.to_csv(path + "//" + classifier + "//" + output_file_name, index=False)

    # aggregate the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()

    return res


def compare_results(path_raw_data_file, path_ta_dir):
    res_raw_data = pd.read_csv(path_raw_data_file, sep=',', header=0, encoding="utf-8")

    df = pd.DataFrame(columns=["classifier_name", "archive_name", "precision improvement",
                               "accuracy improvement", "recall improvement", "equal precision",
                               "equal accuracy", "equal recall", "method", "nb bins", "paa", "std",
                               "max gap", "gradient_window_size"])
    measures = {"precision": None, "accuracy": None, "recall": None}
    equals_measures = {"equal_precision": None, "equal_accuracy": None, "equal_recall": None}

    for root, dirs, files in os.walk(path_ta_dir):
        for file in files:
            if file.endswith(".csv"):

                arguments = re.split('[_.]', file)

                res_ta_data = pd.read_csv(root + "//" + file, sep=',', header=0, encoding="utf-8")

                classifiers = pd.unique(res_ta_data["classifier_name"])
                archives = pd.unique(res_ta_data["archive_name"])

                for classifier in classifiers:
                    for archive in archives:
                        ta_data = res_ta_data.loc[res_ta_data["classifier_name"].eq(classifier) &
                                                  res_ta_data["archive_name"].eq(archive)]
                        raw_data = res_raw_data.loc[res_raw_data["classifier_name"].eq(classifier) &
                                                    res_raw_data["archive_name"].eq(archive)]

                        if ta_data.shape[0] == raw_data.shape[0]:
                            for measure in measures:
                                cmp = pd.to_numeric(res_ta_data[measure]) - pd.to_numeric(res_raw_data[measure])
                                measures[measure] = (cmp[cmp > 0].count() / ta_data.shape[0]) * 100
                                equals_measures["equal_" + measure] = (cmp[cmp == 0].count() / ta_data.shape[0]) * 100

                            new_row = pd.Series(data={"classifier_name": classifier, "archive_name": archive,
                                                      "precision improvement": measures['precision'],
                                                      "accuracy improvement": measures["accuracy"],
                                                      "recall improvement": measures["recall"],
                                                      "equal recall": equals_measures["equal_recall"],
                                                      "equal precision": equals_measures["equal_accuracy"],
                                                      "equal accuracy": equals_measures["equal_accuracy"],
                                                      "method": arguments[1], "nb bins": arguments[2],
                                                      "paa": arguments[3], "std": arguments[4],
                                                      "max gap": arguments[5],
                                                      "gradient_window_size": arguments[6]})
                            df = df.append(new_row, ignore_index=True)

                        else:
                            print("The two files contains different number of datasets")

    config = ConfigClass()
    if os.path.exists(config.get_path() + '//results.csv'):
        df.to_csv(config.get_path() + "//results.csv", index=False, mode='a', header=0)
    else:
        df.to_csv(config.get_path() +'//results.csv', index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1


def viz_perf_themes(root_dir, df):
    df_themes = df.copy()
    themes_index = []
    # add the themes
    for dataset_name in df.index:
        themes_index.append(utils_folder.constants.dataset_types[dataset_name])

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum(axis=1)
    df_themes['#'] = themes_counts

    for classifier in CLASSIFIERS:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')


def viz_perf_train_size(root_dir, df):
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')


def viz_perf_classes(root_dir, df):
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)
    df_classes = df_classes.groupby(level=0, axis=0).mean()
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')


def viz_perf_length(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')


def viz_plot(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCRArchive_2018')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for index, dataset_name in df.iloc[0].items():
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0, xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir + 'plot.pdf')


def viz_for_survey_paper(root_dir, filename='results-ucr-mts.csv'):
    df = pd.read_csv(root_dir + filename, encoding="utf-8")
    df = df.T
    df = df.round(decimals=2)

    # get table performance per themes
    # viz_perf_themes(root_dir,df)

    # get table performance per train size
    # viz_perf_train_size(root_dir,df)

    # get table performance per classes
    # viz_perf_classes(root_dir,df)

    # get table performance per length
    # viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir, df)


def viz_cam(root_dir):
    import tensorflow.keras as keras
    import sklearn
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'

    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    max_length = 2000
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count = 0
        c_x_train = x_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                if dataset_name == 'Gun_Point':
                    if c == 1:
                        plt.yticks([-1.0, 0.0, 1.0, 2.0])
                    else:
                        plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png',
                    bbox_inches='tight', dpi=1080)


def merge_two_columns(x):
    # merged = str(round(x[3] * 100, 1)) + "(" + str(round(x[4] * 100, 1)) + ")"
    merged = round(x[4] * 100, 1)
    return merged


def merge_accuracy(x):
    merged = str(round(x.accuracy, 2)) + " / " + str(round(x.accuracy_before, 2)) + " (" + str(round(
        x.accuracy - x.accuracy_before, 2)) + ")"
    return merged

# todo
def create_results_table():
    file_path = "C:\\Users\\Shaha\\Desktop\\results-mts.csv"

    df = pd.read_csv(file_path, encoding="utf-8")

    res_df = df.groupby(["classifier_name", "archive_name", "dataset_name"], as_index=False).agg({"accuracy": [np.mean,
                                                                                                               np.std]})
    res_df["res"] = res_df.apply(merge_two_columns, axis=1)

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"]

    mts_df = mts_df.pivot(index="dataset_name", columns="classifier_name", values="res").reset_index()
    ucr_df = ucr_df.pivot(index="dataset_name", columns="classifier_name", values="res").reset_index()

    mts_df.to_csv("mts_results.csv", index=False)
    ucr_df.to_csv("ucr_results.csv", index=False)


def results_table_by_dataset_lengths(root_dir, path_file_raw_data, path_file_data_ta):
    res_df = set_df_for_graphs_and_tables(path_file_data_ta, path_file_raw_data)

    res_df["res"] = res_df.apply(merge_accuracy, axis=1)

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"]

    mts_df = mts_df.pivot(index="groups_lengths", columns="classifier_name", values="res").reset_index()
    ucr_df = ucr_df.pivot(index="groups_lengths", columns="classifier_name", values="res").reset_index()

    mts_df.to_csv(root_dir + "/Results/mts_archive/results_by_ds_length.csv", index=False)
    ucr_df.to_csv(root_dir + "/Results/UCRArchive_2018/results_by_ds_length.csv", index=False)


def set_df_for_graphs_and_tables(path_file_data_ta, path_file_raw_data):
    raw_data_df = pd.read_csv(path_file_raw_data, encoding="utf-8")
    data_after_ta_df = pd.read_csv(path_file_data_ta, encoding="utf-8")

    raw_data_df = raw_data_df.rename(columns={"accuracy": "accuracy_before"})

    df = pd.merge(raw_data_df, data_after_ta_df, on=["dataset_name", "classifier_name", "archive_name", "iteration"])

    ucr_length = open_pickle("uts_length")
    mts_length = open_pickle("mts_length")
    lengths = {**ucr_length, **mts_length}
    length_df = pd.DataFrame(list(lengths.items()), columns=['dataset_name', 'lengths'])

    merged = pd.merge(df, length_df, on="dataset_name")

    merged.loc[merged.lengths < 81, 'groups_lengths'] = "< 81"
    merged.loc[(merged.lengths >= 81) & (merged.lengths <= 250), 'groups_lengths'] = "81 - 250"
    merged.loc[(merged.lengths >= 251) & (merged.lengths <= 450), 'groups_lengths'] = "251 - 450"
    merged.loc[(merged.lengths >= 451) & (merged.lengths <= 700), 'groups_lengths'] = "451 - 700"
    merged.loc[(merged.lengths >= 701) & (merged.lengths <= 1000), 'groups_lengths'] = "701 - 1000"
    merged.loc[merged.lengths > 1000, 'groups_lengths'] = " > 1000"

    res_df = merged.groupby(["classifier_name", "archive_name", "groups_lengths"], as_index=False).agg \
        ({"accuracy": np.mean, "accuracy_before": np.mean})
    return res_df


def create_df_for_rank_graph(path):
    df = pd.read_csv(path, encoding="utf-8")

    res_df = df.groupby(["classifier_name", "archive_name", "dataset_name"], as_index=False).agg({"accuracy": np.mean})

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"]

    mts_df.drop("archive_name", axis=1, inplace=True)
    ucr_df.drop("archive_name", axis=1, inplace=True)

    draw_cd_diagram(df_perf=mts_df, title='Accuracy', labels=True)
    draw_cd_diagram(df_perf=ucr_df, title='Accuracy', labels=True)


def create_graphs(root_dir, path_before, path_after):
    res_df = set_df_for_graphs_and_tables(path_before, path_after)

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"].iloc[:: -1]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"].iloc[:: -1]

    for classifier in CLASSIFIERS:
        ucr_classifier = ucr_df.loc[ucr_df.classifier_name == classifier]
        mts_classifier = mts_df.loc[mts_df.classifier_name == classifier]

        sns.set_theme(style="darkgrid")
        colors = ["#FF0B04", "#4374B3"]
        sns.set_palette(sns.color_palette(colors))

        # sns.set_palette(sns.color_palette("viridis", as_cmap=True))

        sns.lineplot(x="groups_lengths", y="accuracy", data=ucr_classifier)
        fig = sns.lineplot(x="groups_lengths", y="accuracy_before", data=ucr_classifier)
        fig.set(xlabel='Length', ylabel='Accuracy')
        fig.legend(['Accuracy Before TA', 'Accuracy After TA'])

        plt.savefig("Plots/UCR/" + classifier + ".png")
        plt.clf()

        sns.lineplot(x="groups_lengths", y="accuracy", data=mts_classifier)
        fig = sns.lineplot(x="groups_lengths", y="accuracy_before", data=mts_classifier)
        fig.set(xlabel='Length', ylabel='Accuracy')

        plt.savefig("Plots/MTS/" + classifier + ".png")
        plt.clf()
        print()




