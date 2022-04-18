import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os, time
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from utils_folder.configuration import ConfigClass
from utils_folder.ranking_graph import draw_cd_diagram

from builtins import print

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, \
    matthews_corrcoef, cohen_kappa_score

from scipy.interpolate import interp1d
from scipy.io import loadmat

matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

config = ConfigClass()
config.set_seed()


def open_pickle(name):
    file = open(config.path + "/Project/temporal_abstraction_f/pickle_files//" + name + ".pkl", "rb")
    data = pickle.load(file)
    return data


def write_pickle(name, data):
    file = open(config.path + "/Project/temporal_abstraction_f/pickle_files//" + name + ".pkl", "wb")
    pickle.dump(data, file)
    file.close()


def check_pickle_exists(name):
    return os.path.exists(config.path + "/Project/temporal_abstraction_f/pickle_files//" + name + ".pkl")


def is_locked(filepath, cli, sep):
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            if cli:
                file_object = pd.read_csv(filepath)

            # Read all datasets
            else:
                if config.archive == "UCR" and not config.combination:
                    file_object = pd.read_csv(filepath, sep=sep, header=None)
                # MTS
                else:
                    file_object = np.load(filepath)

            if file_object is not None:
                locked = False
        except IOError as message:
            locked = True

    return locked, file_object


def wait_for_files(filepath, cli=False, sep=','):
    flag, data = is_locked(filepath, cli, sep)
    while flag:
        flag, data = is_locked(filepath, cli, sep)
        time.sleep(random.randint(0, 20))

    return data


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path


def read_all_datasets(config):
    datasets_dict = {}

    if config.archive == 'MTS':
        datasets = config.MTS_DATASET_NAMES
        path = config.mts_path
    else:
        datasets = config.UNIVARIATE_DATASET_NAMES_2018
        path = config.ucr_path

    for dataset_name in datasets:
        root_dir_dataset = path + '/' + dataset_name + '/'

        if config.afterTA:
            if config.combination:
                x_train = wait_for_files(config.path_transformation2 + dataset_name + "//type" +
                                         config.transformation_number + '_train_combination.npy')
                x_test = wait_for_files(config.path_transformation2 + dataset_name + "//type" +
                                        config.transformation_number + '_test_combination.npy')
            else:
                x_train = wait_for_files(config.path_transformation2 + dataset_name + "//type" +
                                         config.transformation_number + '_train.npy')
                x_test = wait_for_files(config.path_transformation2 + dataset_name + "//type" +
                                        config.transformation_number + '_test.npy')

            y_train = wait_for_files(root_dir_dataset + 'y_train.npy')
            y_test = wait_for_files(root_dir_dataset + 'y_test.npy')

        # Raw data
        else:
            if config.archive == "MTS":
                x_train = wait_for_files(root_dir_dataset + 'x_train.npy')
                y_train = wait_for_files(root_dir_dataset + 'y_train.npy')
                x_test = wait_for_files(root_dir_dataset + 'x_test.npy')
                y_test = wait_for_files(root_dir_dataset + 'y_test.npy')

            else:
                root_dir_dataset = config.ucr_path + '/' + dataset_name + '/'

                df_train = wait_for_files(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t')
                df_test = wait_for_files(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t')

                # Padding missing values
                if df_train.isnull().sum().sum() != 0:
                    df_train = df_train.interpolate(method='linear', limit_direction='both', axis=1)

                if df_test.isnull().sum().sum() != 0:
                    df_test = df_test.interpolate(method='linear', limit_direction='both', axis=1)

                y_train = df_train.values[:, 0]
                y_test = df_test.values[:, 0]

                x_train = df_train.drop(columns=[0])
                x_test = df_test.drop(columns=[0])

                x_train.columns = range(x_train.shape[1])
                x_test.columns = range(x_test.shape[1])

                x_train = x_train.values
                x_test = x_test.values

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())

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
    mts_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each entity
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
            mts_x[i, :, j] = new_ts

    return mts_x


def transform_mts_to_ucr_format():
    mts_dict = {}
    length_dict = {}

    for dataset_name in config.MTS_DATASET_NAMES:
        out_dir = config.mts_path + dataset_name + '/'

        a = loadmat(config.mts_path + dataset_name + '/' + dataset_name + '.mat')
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

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        mts_dict[dataset_name] = {"time_serious_length": max_length, "number_of_entities_train": len(x_train),
                                  "number_of_entities_test": len(x_test), "number_of_attributes": n_var}

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()

        x_train = transform_to_same_length(x_train, n_var, max_length)
        x_test = transform_to_same_length(x_test, n_var, max_length)

        length_dict[dataset_name] = max_length

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')

    write_pickle("MTS_Dictionary", mts_dict)
    write_pickle("length_dict", length_dict)


def calculate_metrics(y_true, y_pred, learning_time, predicting_time, y_true_val=None, y_pred_val=None,y_pred_new =None):
    res = pd.DataFrame(data=np.zeros((1, 11), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'mcc', 'cohen_kappa', 'learning_time',
                                'predicting_time', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted','auc'])

    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['recall'] = recall_score(y_true, y_pred, average='macro')

    res['accuracy'] = balanced_accuracy_score(y_true, y_pred)

    res['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
    res['f1_score_micro'] = f1_score(y_true, y_pred, average='micro')
    res['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted')

    # Matthews correlation coefficient
    res['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Cohen’s kappa

    if len(set(y_true + y_pred)) == 1:
        res['cohen_kappa'] = 1
    else:
        res['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    #res['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)


    # This is useful when transfer learning is used with cross validation
    if y_true_val is not None:
        res['accuracy_val'] = balanced_accuracy_score(y_true_val, y_pred_val)

    # AUC
    #y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    if y_pred_new is None:
        res['auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr')
    else:
        res['auc'] = roc_auc_score(y_true, y_pred_new, multi_class='ovr')  # todo - think if ovo / ovr/ raise


    res['learning_time'] = learning_time
    res['predicting_time'] = predicting_time

    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0], columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def generate_results_csv(config, params):
    res = pd.DataFrame(data=np.zeros((0, 14), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name', 'precision', 'recall', 'accuracy',
                                'mcc', "cohen_kappa", "f1_score_macro", "f1_score_micro",
                                "f1_score_weighted", 'learning_time', 'predicting_time','auc'])

    dataset_list = config.UNIVARIATE_DATASET_NAMES_2018 if config.archive == "UCR" else config.MTS_DATASET_NAMES

    for it in range(config.ITERATIONS):
        for dataset_name in dataset_list:
            output_dir = config.path + "/ResultsProject//DNN//" + config.archive + "//" + config.classifier + '/' + \
                         config.method + "/" + params + "//itr" + str(it) + '//' + dataset_name + '/' + \
                         'df_metrics.csv'

            if not os.path.exists(output_dir):
                continue

            df_metrics = pd.read_csv(output_dir)
            df_metrics['classifier_name'] = config.classifier
            df_metrics['archive_name'] = config.archive
            df_metrics['dataset_name'] = dataset_name
            df_metrics['iteration'] = it
            res = pd.concat((res, df_metrics), axis=0, sort=False)

    path = config.path + "ResultsProject//AfterTA//" if config.afterTA else config.path + "ResultsProject//RawData//"
    path += config.archive + "//" + config.classifier + "//" + config.method + "//"

    create_directory(path)
    res.to_csv(path + params + ".csv", index=False)


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


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, learning_time, predicting_time):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, learning_time, predicting_time)

    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_metrics['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, learning_time, predicting_time,y_pred_new=None,lr=True, y_true_val=None,
              y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, learning_time, predicting_time, y_true_val, y_pred_val,y_pred_new)

    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']

    df_metrics['best_model_nb_epoch'] = index_best_model
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    if lr:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def merge_two_columns(x):
    return round(x[4] * 100, 1)


def merge_accuracy(x):
    merged = str(round(x.accuracy, 2)) + " / " + str(round(x.accuracy_before, 2)) + " (" + str(round(
        x.accuracy - x.accuracy_before, 2)) + ")"
    return merged


def create_df_for_rank_graph(path):
    df = pd.read_csv(path, encoding="utf-8")

    res_df = df.groupby(["classifier_name", "archive_name", "dataset_name"], as_index=False).agg({"accuracy": np.mean})

    ucr_df = res_df.loc[res_df.archive_name == "UCRArchive_2018"]
    mts_df = res_df.loc[res_df.archive_name == "mts_archive"]

    mts_df.drop("archive_name", axis=1, inplace=True)
    ucr_df.drop("archive_name", axis=1, inplace=True)

    draw_cd_diagram(df_perf=mts_df, title='Accuracy', labels=True)
