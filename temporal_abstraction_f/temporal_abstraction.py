import sys
sys.path.append('../')

from configuration import ConfigClass
from multivariate_ta_1 import MultivariateTA1
from univariate_ta_1 import UnivariateTA1
from set_parameters import create_three_files
from multivariate_ta_2 import new_mts_files
from univariate_ta_2 import new_uts_files

from utils_folder.utils import write_pickle

from os import path
sys.path.insert(0, '../../Hugobot')
from cli import run_cli

from utils_folder.constants import MTS_DATASET_NAMES
from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018


def run():
    config = ConfigClass()

    # Make the first temporal abstraction -> original data sets to hugobot format
    # print("Step 1: transformation 1")
    #
    # print("\t Univariate")
    # uni_ta_1 = UnivariateTA1(config.get_ucr_path(), 0)
    # next_attribute = uni_ta_1.convert_all_UTS()

    # print("\t Multivariate")
    # multi_ta_1 = MultivariateTA1(config.get_mts_path(), next_attribute)
    # next_attribute = multi_ta_1.convert_all_MTS()
    #
    # write_pickle("next_property_index", {"ID": next_attribute})

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    # print("Step 2: make the gkb.csv, ta.csv and ppa.csv")
    # create_three_files(path=config.get_prop_path(),
    #                    method=config.get_method(),
    #                    nb_bins=config.get_nb_bin(),
    #                    paa_window_size=config.get_paa_window_size(),
    #                    std_coefficient=config.get_std_coefficient(),
    #                    max_gap=config.get_max_gap())

    # print("Step 3: run hugobot")
    # print("\t Multivariate")
    # run_cli(config.get_prop_path(), config.get_mts_path(), MTS_DATASET_NAMES, "mtsdata")

    # print("\t Univariate")
    # run_cli(config.get_prop_path(), config.get_ucr_path(), DATASET_NAMES_2018, "UCRArchive_2018")

    # Make the second temporal abstraction -> hugobot output files to original format
    # print("Step 4: transformation 2")
    # print("\t Multivariate")
    # new_mts_files(config.get_mts_path())

    print("\t Univariate")
    new_uts_files(config.get_ucr_path())


run()
