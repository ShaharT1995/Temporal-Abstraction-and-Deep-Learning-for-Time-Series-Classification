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


def run():
    config = ConfigClass()

    # Make the first temporal abstraction -> original data sets to hugobot format
    uni_ta_1 = UnivariateTA1(config.get_ucr_path(), 0)
    next_attribute = uni_ta_1.convert_all_UTS()

    multi_ta_1 = MultivariateTA1(config.get_mts_path(), next_attribute)
    next_attribute = multi_ta_1.convert_all_MTS()

    write_pickle("next_property_index", {"ID": next_attribute})

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    create_three_files(path=config.get_prop_path(),
                       method=config.get_method(),
                       nb_bins=config.get_nb_bin(),
                       paa_window_size=config.get_paa_window_size(),
                       std_coefficient=config.get_std_coefficient(),
                       max_gap=config.get_max_gap())

    run_cli(config.get_prop_path(), config.get_mts_path(), MTS_DATASET_NAMES)

    # Make the second temporal abstraction -> hugobot output files to original format
    new_mts_files(config.get_mts_path())
    new_uts_files(config.get_ucr_path())


run()
