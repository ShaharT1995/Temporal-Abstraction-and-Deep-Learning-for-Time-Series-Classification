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
    #next_attribute = open_pickle("next_property_index" + config.archive)["ID"]
    attribute_dict= open_pickle("attributes_dict_ucr") if config.archive == "UCR" else open_pickle("attributes_dict_mts")
    dataset_list = config.UNIVARIATE_DATASET_NAMES_2018 if config.archive == "UCR" else config.MTS_DATASET_NAMES
    for ds in dataset_list:
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
                data={"TemporalPropertyID": property_id, "PAAWindowSize": paa_window_size, "StdCoefficient": std_coefficient,
                      "MaxGap": max_gap})
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
            df_gkb.to_csv(path + ds +'//gkb.csv', index=False)
        # Save all data frames to csv
        df_ta.to_csv(path + ds + '//ta.csv', index=False)
        df_pp.to_csv(path + ds + '//pp.csv', index=False)
