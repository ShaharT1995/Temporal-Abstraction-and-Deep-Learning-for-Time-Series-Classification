

import os
import sys

import numpy as np

import main

# todo
sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/SyncProject/')

from utils_folder.configuration import ConfigClass
from temporal_abstraction_f.multivariate_ta_1 import MultivariateTA1
from temporal_abstraction_f.univariate_ta_1 import UnivariateTA1
from temporal_abstraction_f.set_parameters import create_three_files
from temporal_abstraction_f.multivariate_ta_2 import new_mts_files
from temporal_abstraction_f.univariate_ta_2 import new_uts_files

from utils_folder.utils import write_pickle, open_pickle

# todo
sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/SyncProject/Hugobot')

from Hugobot.cli import run_cli

from utils_folder.utils import generate_results_csv
from utils_folder.constants import MTS_DATASET_NAMES
from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils_folder.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils_folder.constants import CLASSIFIERS

from utils_folder.utils import compare_results

import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

# Setting the graph-level random seed.
tf.random.set_seed(89)

from tensorflow.python.keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


def run():
    config = ConfigClass()
    prop_path = config.get_prop_path() + ARCHIVE_NAMES[0] + "//" + CLASSIFIERS[0] + "//" + config.get_method()[0] + "//"
    if not os.path.exists(prop_path):
        os.makedirs(prop_path)

    # Make the first temporal abstraction -> original data sets to hugobot format
    # print("Step 1: transformation 1")
    #
    # print("\t Univariate")
    # uni_ta_1 = UnivariateTA1(config.get_ucr_path(), 0)
    # next_attribute, attributes_dict = uni_ta_1.convert_all_UTS()
    #
    # print("\t Multivariate")
    # multi_ta_1 = MultivariateTA1(config.get_mts_path(), next_attribute, attributes_dict)
    # next_attribute = multi_ta_1.convert_all_MTS()
    #
    # write_pickle("next_property_index", {"ID": next_attribute})
    # print()
    #
    # # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    # print("Step 2: make the gkb.csv, ta.csv and ppa.csv")
    # print()

    running_dict = open_pickle("running_dict")

    print(np.random.get_state()[1][0])
    print(rn.random())

    running_dict = execute_running(config, running_dict, 2, "gradient", 3, 1, 1,
                                   1)

    print(np.random.get_state()[1][0])
    print(rn.random())
    running_dict = execute_running(config, running_dict, 2, "gradient", 3, 1, 1,
                                   5)

    # for method in config.get_method():
    #     for nb_bin in config.get_nb_bin():
    #         for paa in config.get_paa_window_size():
    #             for std in config.get_std_coefficient():
    #                 for max_gap in config.get_max_gap():
    #                     if method == "gradient":
    #                         for gradient_window in config.get_gradient_window_size():
    #                             running_dict = execute_running(config, running_dict, max_gap, method, nb_bin, paa, std,
    #                                                            gradient_window)
    #                     else:
    #                         running_dict = execute_running(config, running_dict, max_gap, method, nb_bin, paa, std)

    print("Compare the results between running on the raw data and the data after the TA")
    path_raw_data_file = config.get_path() + "Results//raw_data_results.csv"
    path_ta_dir = config.get_path() + "//Results//ResultsAfterTA"

    compare_results(path_raw_data_file, path_ta_dir)
    print("Done comparing")


def execute_running(config, running_dict, max_gap, method, nb_bin, paa, std, gradient_window=None):
    # todo - Add gradient to the print
    print("-------------------------------------------------------------------------------------")
    print("Method: " + method + ", Bins: " + str(nb_bin) + ", PAA: " + str(paa) + ", STD: " +
          str(std) + ", Max_Gap: " + str(max_gap))
    print("-------------------------------------------------------------------------------------")
    print()

    key = (ARCHIVE_NAMES[0], CLASSIFIERS[0], method, nb_bin, paa, std, max_gap, gradient_window)
    # if key in running_dict:
    #     # print("Already Done!")
    #     # print()
    #
    #     return running_dict
    #
    # else:
    prop_path = config.get_prop_path() + ARCHIVE_NAMES[0] + "//" + CLASSIFIERS[0] + "//" + method + "//"
    if not os.path.exists(prop_path):
        os.makedirs(prop_path)
    #
    # create_three_files(path=prop_path,
    #                    method=method,
    #                    nb_bins=nb_bin,
    #                    paa_window_size=paa,
    #                    std_coefficient=std,
    #                    max_gap=max_gap,
    #                    gradient_window_size=gradient_window)
    #
    # print("Step 3: run hugobot")
    # print("\tMultivariate")
    # run_cli(prop_path, MTS_DATASET_NAMES, "mtsdata", method, max_gap)
    #
    # print("\tUnivariate")
    # run_cli(prop_path, DATASET_NAMES_2018, "UCRArchive_2018", method, max_gap)
    #
    # # Make the second temporal abstraction -> hugobot output files to original format
    # print("Step 4: transformation 2")
    # print("\tMultivariate")
    # new_mts_files(config.get_mts_path())
    #
    # print("\tUnivariate")
    # new_uts_files(config.get_ucr_path())

    # print("Step 5: Run all:")
    params = "res_" + str(method) + "_" + str(nb_bin) + "_" + str(paa) + "_" + str(std) \
                    + "_" + str(max_gap) + "_" + str(gradient_window)
    main.run_all(params)
    print("")

    print("Step 6: Generate Results to CSV")

    for classifier in CLASSIFIERS:
        file_name = "res_" + str(method) + "_" + str(nb_bin) + "_" + str(paa) + "_" + str(std) \
                    + "_" + str(max_gap) + "_" + str(gradient_window) + ".csv"
        generate_results_csv(file_name, config.get_path(), classifier, params, False)

    running_dict[key] = True

    write_pickle("running_dict", running_dict)
    return running_dict

run()

# univariate_dict = open_pickle("univariate_dict")
# print(univariate_dict[('SmoothSubspace', 'train')]["classes"])

# file_path = "/home/shaharap/results-ucr-mts.csv"
# file_path1 = "/home/shaharap/test.csv"

# # results_table_by_dataset_lengths("/home/shaharap/", file_path, file_path)
#
# create_df_for_rank_graph(file_path)
# print("SHAHAR")
# config = ConfigClass()
# create_graphs(config.get_path(), file_path, file_path1)
