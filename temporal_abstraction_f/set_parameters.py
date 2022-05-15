from pathlib import Path

import pandas as pd
from utils_folder.utils import open_pickle, create_directory


def create_three_files(config, path, method, nb_bins, paa_window_size, std_coefficient, max_gap,
                       gradient_window_size=None):
    """
    :param path: the location of the database
    :param method: the temporal abstraction method
    :param nb_bins: number of bins
    :param paa_window_size: the window size of the paa
    :param std_coefficient: outliers remover (default std_coefficient=-1 (not perform))
    :param max_gap: the max gap between two values
    :param gradient_window_size: the window size of the gradient method
    :return:
    """

    attributes_dict_addition = ""
    if config.perEntity:
        attributes_dict_addition = "_per_entity"

    if config.archive == "UCR":
        attribute_dict = open_pickle("attributes_dict_ucr" + attributes_dict_addition)
        dataset_list = config.UNIVARIATE_DATASET_NAMES_2018
    else:
        attribute_dict = open_pickle("attributes_dict_mts" + attributes_dict_addition)
        dataset_list = config.MTS_DATASET_NAMES

    for ds in dataset_list:

        ta_file = Path(path + ds + '//ta.csv')
        if ta_file.is_file():
            print(ds)
            print("\tFiles already exists, continue to the next step")
            continue

        # GKB
        index = 0
        df_gkb = pd.DataFrame(
            columns=["StateID", "TemporalPropertyID", "Method", "BinID", "BinLow", "BinHigh", "BinLowScore"])

        bin_degree = 180 / nb_bins

        # TA
        df_ta = pd.DataFrame(columns=["TemporalPropertyID", "Method", "NbBins", "GradientWindowSize"])

        # PP
        df_pp = pd.DataFrame(columns=["TemporalPropertyID", "PAAWindowSize", "StdCoefficient", "MaxGap"])

        for property_id in attribute_dict[ds]:
            # TA
            new_row = pd.Series(data={"TemporalPropertyID": property_id, "Method": method, "NbBins": nb_bins,
                                      "GradientWindowSize": gradient_window_size})
            df_ta = df_ta.append(new_row, ignore_index=True)

            # PP
            new_row = pd.Series(
                data={"TemporalPropertyID": property_id, "PAAWindowSize": paa_window_size, "MaxGap": max_gap,
                      "StdCoefficient": std_coefficient})
            df_pp = df_pp.append(new_row, ignore_index=True)

            if method == "gradient":
                bin_low = -90

                for bin_id in range(nb_bins):
                    new_row = pd.Series(data={"StateID": index, "TemporalPropertyID": property_id, "Method": "gradient",
                                              "BinID": bin_id, "BinLow": bin_low, "BinHigh": bin_low + bin_degree,
                                              "BinLowScore": ""})
                    bin_low += bin_degree
                    index += 1

                    df_gkb = df_gkb.append(new_row, ignore_index=True)

        create_directory(path + ds)
        if method == "gradient":
            df_gkb.to_csv(path + ds + '//gkb.csv', index=False)
        # Save all data frames to csv
        df_ta.to_csv(path + ds + '//ta.csv', index=False)
        df_pp.to_csv(path + ds + '//pp.csv', index=False)
