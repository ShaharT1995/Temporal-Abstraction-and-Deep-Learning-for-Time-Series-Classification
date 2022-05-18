import os
import numpy as np
import pandas as pd
import sklearn
from utils_folder.utils import read_all_datasets, create_directory


def fit_classifier(config, iter, datasets_dict, dataset_name, classifier_name, output_directory):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # Transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # Save the original y, because later we will use binary
    y_test_true = np.argmax(y_test, axis=1)

    # If univariate add a dimension to make it multivariate with one dimension
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(config, classifier_name, input_shape, nb_classes, output_directory,
                                   len(pd.unique(y_test_true)))

    classifier.fit(x_train, y_train, x_test, y_test, y_test_true, iter)


def create_classifier(config, classifier_name, input_shape, nb_classes, output_directory, cv=10, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)

    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)

    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)

    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'rocket':
        if config.archive == "UCR" and config.transformation_number == "1" and config.combination is False:
            from classifiers import rocket_ucr
            return rocket_ucr.Classifier_Rocket(output_directory, verbose)
        else:
            from classifiers import rocket_mts
            return rocket_mts.RocketClassifier(output_directory, cv)

    if classifier_name == 'lstm_fcn':
        from classifiers import lstm_fcn
        return lstm_fcn.Classifier_LSTMFCN(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'mlstm_fcn':
        from classifiers import mlstm_fcn
        return mlstm_fcn.Classifier_MLSTM_FCN(output_directory, input_shape, nb_classes, verbose)


def run_all(config, params):
    config.set_seed()

    datasets_dict = read_all_datasets(config)

    print("Classifier: " + config.classifier)

    for iter in range(config.ITERATIONS):
        print('\titer', iter)

        dataset_list = config.UNIVARIATE_DATASET_NAMES_2018 if config.archive == "UCR" else config.MTS_DATASET_NAMES
        for dataset_name in dataset_list:
            print('\t\tdataset_name: ', dataset_name)

            if dataset_name == "CMUsubject16" and (config.classifier == "mlp" or config.classifier == "mcdcnn"):
                print('\t\tMemory Error in : ', dataset_name)
                continue

            output_directory = config.path + "/ResultsProject//DNN//" + config.archive + "//" + config.classifier + '/' + \
                               config.method + "/" + params + "//itr" + str(iter) + '//' + dataset_name + '/'

            if os.path.exists(output_directory + "/DONE"):
                print("\t\t\tAlready Done")
                continue

            create_directory(output_directory)

            fit_classifier(config, iter, datasets_dict, dataset_name, config.classifier, output_directory)

            print('\t\t\tDONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')
