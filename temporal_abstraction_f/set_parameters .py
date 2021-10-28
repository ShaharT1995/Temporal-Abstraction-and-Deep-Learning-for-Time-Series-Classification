import pandas as pd
from utils.constants import NEXT_ATTRIBUTE_ID


def create_three_files(path, method, nb_bins, paa_window_size, std_coefficient, max_gap, gradient_window_size=None):
    # GKB
    index = 0
    df_gkb = pd.DataFrame(
        columns=["StateID", "TemporalPropertyID", "Method", "BinID", "BinLow", "BinHigh", "BinLowScore"])

    bin_degree = 180 / nb_bins

    # TA
    df_ta = pd.DataFrame(columns=["TemporalPropertyID", "Method", "NbBins", "GradientWindowSize"])

    # PP
    df_pp = pd.DataFrame(columns=["TemporalPropertyID", "PAAWindowSize", "StdCoefficient", "MaxGap"])

    for property_id in range(NEXT_ATTRIBUTE_ID):
        # TA
        new_row = pd.Series(data={"TemporalPropertyID": property_id, "Method": method, "NbBins": nb_bins,
                                  "GradientWindowSize": gradient_window_size})
        df_ta = df_ta.append(new_row, ignore_index=True)

        # PP
        new_row = pd.Series(
            data={"TemporalPropertyID": property_id, "PAAWindowSize": paa_window_size, "StdCoefficient": std_coefficient,
                  "MaxGap": max_gap})
        df_pp = df_pp.append(new_row, ignore_index=True)

        # GKB
        bin_low = -90

        for bin_id in range(nb_bins):
            new_row = pd.Series(data={"StateID": index, "TemporalPropertyID": property_id, "Method": "gradient",
                                      "BinID": bin_id, "BinLow": bin_low, "BinHigh": bin_low + bin_degree,
                                      "BinLowScore": ""})
            bin_low += bin_degree
            index += 1

            df_gkb = df_gkb.append(new_row, ignore_index=True)

    # Save all data frames to csv
    df_ta.to_csv(path + '\\ta.csv', index=False)
    df_pp.to_csv(path + '\\pp.csv', index=False)
    df_gkb.to_csv(path + '\\gkb.csv', index=False)


create_three_files(path="C:\\Users\\Shaha\\Desktop\\TA\\TEST",
                   method="equal-frequency",
                   nb_bins=3,
                   paa_window_size=1,
                   std_coefficient=-1,
                   max_gap=-1)
