def run():
    CLASSIFIERS = ['mcdcnn', 'fcn', 'mlp', 'twiesn', 'cnn']
    METHODS = ['sax', 'gradient', 'equal-frequency', 'equal-width', 'td4c-cosine']

    for classifier in CLASSIFIERS:
        for method in METHODS:
            for nb_bin in config.get_nb_bin():
                for paa in config.get_paa_window_size():
                    for std in config.get_std_coefficient():
                        for max_gap in config.get_max_gap():
                            if method == "gradient":
                                for gradient_window in config.get_gradient_window_size():
                                    params = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
                                             + "" + str(max_gap) + "" + str(gradient_window)

                                    print(params)

                                    file_name = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
                                                + "" + str(max_gap) + "" + str(gradient_window) + ".csv"
                                    generate_results_csv(file_name, config.get_path(), classifier, params, True)
                            else:
                                params = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
                                         + "" + str(max_gap) + "None"

                                print(params)

                                file_name = "res_" + str(method) + "" + str(nb_bin) + "" + str(paa) + "_" + str(std) \
                                            + "" + str(max_gap) + "None.csv"
                                generate_results_csv(file_name, config.get_path(), classifier, params, True)


if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/SyncProject/')

    from utils_folder.configuration import ConfigClass
    config = ConfigClass()

    from temporal_abstraction_f.multivariate_ta_1 import MultivariateTA1
    from temporal_abstraction_f.univariate_ta_1 import UnivariateTA1
    from temporal_abstraction_f.set_parameters import create_three_files
    from temporal_abstraction_f.multivariate_ta_2 import new_mts_files
    from temporal_abstraction_f.univariate_ta_2 import new_ucr_files

    from utils_folder.utils import write_pickle, open_pickle, results_table_by_dataset_lengths, create_graphs, \
    create_df_for_rank_graph

    # todo
    sys.path.insert(0, '/sise/robertmo-group/TA-DL-TSC/SyncProject/Hugobot')

    from Hugobot.cli import run_cli

    from utils_folder.utils import generate_results_csv
    from utils_folder.constants import MTS_DATASET_NAMES
    from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
    from utils_folder.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
    from utils_folder.constants import CLASSIFIERS

    from utils_folder.utils import compare_results

    run()

# univariate_dict = open_pickle("univariate_dict")
# print(univariate_dict[('SmoothSubspace', 'train')]["classes"])

#file_path = '/sise/robertmo-group/TA-DL-TSC/Results/mcdcnn/results.csv'
#file_path1 = "/sise/robertmo-group/TA-DL-TSC//Results//ResultsAfterTA/mcdcnn/res_gradient_2_1_-1_1_1.csv"

# results_table_by_dataset_lengths("/sise/robertmo-group//TA-DL-TSC/", file_path, file_path1)

# create_df_for_rank_graph("C:\\Users\\Shaha\Desktop\\results_new.csv")
# print("SHAHAR")
# config = ConfigClass()
#create_graphs(config.get_path(), file_path, file_path1)
