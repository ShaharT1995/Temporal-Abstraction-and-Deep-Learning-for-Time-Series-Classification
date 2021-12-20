import os
import pickle
import sys


sys.path.append('../')

from configuration import ConfigClass
from multivariate_ta_1 import MultivariateTA1
from univariate_ta_1 import UnivariateTA1
from set_parameters import create_three_files
from multivariate_ta_2 import new_mts_files
from univariate_ta_2 import new_uts_files

from utils_folder.utils import write_pickle

sys.path.insert(0, '../../Hugobot')
from cli import run_cli

sys.path.insert(0, '../-dl-4-tsc')

from utils_folder.utils import generate_results_csv
from utils_folder.constants import MTS_DATASET_NAMES
from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils_folder.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils_folder.utils import open_pickle

from utils_folder.utils import compare_results


def run():
    config = ConfigClass()

    # Make the first temporal abstraction -> original data sets to hugobot format
    print("Step 1: transformation 1")

    print("\t Univariate")
    uni_ta_1 = UnivariateTA1(config.get_ucr_path(), 0)
    next_attribute, attributes_dict = uni_ta_1.convert_all_UTS()

    print("\t Multivariate")
    multi_ta_1 = MultivariateTA1(config.get_mts_path(), next_attribute, attributes_dict)
    next_attribute = multi_ta_1.convert_all_MTS()

    write_pickle("next_property_index", {"ID": next_attribute})
    # Make the 3 files - gkb.csv, ta.csv and ppa.csv

    print("Step 2: make the gkb.csv, ta.csv and ppa.csv")
    print()
    for method in config.get_method():
        for nb_bin in config.get_nb_bin():
            for paa in config.get_paa_window_size():
                for std in config.get_std_coefficient():
                    for max_gap in config.get_max_gap():
                        print("-------------------------------------------------------------------------------------")
                        print("Method: " + method + ", Bins: " + str(nb_bin) + ", PAA: " + str(paa) + ", STD: " +
                              str(std) + ", Max_Gap: " + str(max_gap))
                        print("-------------------------------------------------------------------------------------")
                        print()

                        create_three_files(path=config.get_prop_path(),
                                           method=method,
                                           nb_bins=nb_bin,
                                           paa_window_size=paa,
                                           std_coefficient=std,
                                           max_gap=max_gap)

                        print("Step 3: run hugobot")
                        print("\tMultivariate")
                        run_cli(config.get_prop_path(), config.get_mts_path(), MTS_DATASET_NAMES, "mtsdata", max_gap)

                        print("\tUnivariate")
                        run_cli(config.get_prop_path(), config.get_ucr_path(), DATASET_NAMES_2018, "UCRArchive_2018",
                                max_gap)

                        # Make the second temporal abstraction -> hugobot output files to original format
                        print("Step 4: transformation 2")
                        print("\tMultivariate")
                        new_mts_files(config.get_mts_path())

                        print("\tUnivariate")
                        new_uts_files(config.get_ucr_path())

                        print("Run all:")
                        os.system('py D:\\Git\\-dl-4-tsc\\main.py run_all')
                        print("")

                        print("Generate Results to CSV")
                        file_name = "res_" + str(method) + "_" + str(nb_bin) + "_" + str(paa) + "_" + str(std)\
                                    + "_" + str(max_gap) + ".csv"

                        generate_results_csv(file_name, config.get_path())

    for archive_name in ARCHIVE_NAMES:
        path_raw_data_file = config.get_path() + archive_name + "\\results\\raw_data_results.csv"
        # path_raw_data_file = "C:\\Users\\Shaha\\Desktop\\1.csv"

        path_ta_dir = config.get_path() + archive_name + "\\results_after_ta\\"
        # path_ta_dir = "C:\\Users\\Shaha\\Desktop\\results_after_ta"
        compare_results(path_raw_data_file, path_ta_dir)


run()

