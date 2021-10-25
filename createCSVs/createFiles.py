import pandas as pd


# Create TA file
def create_ta_file(path, num_of_ds, method, nb_bins, gradient_window_size=None):
    df = pd.DataFrame(columns=["TemporalPropertyID", "Method", "NbBins", "GradientWindowSize"])
    for ds in range(num_of_ds):
        new_row = pd.Series(data={"TemporalPropertyID": ds, "Method": method, "NbBins": nb_bins,
                                    "GradientWindowSize": gradient_window_size})
        df = df.append(new_row, ignore_index=True)

    df.to_csv(path + '\\ta.csv', index=False)


# Create PP file
def create_pp_file(path, num_of_ds, paa_window_size, std_coefficient, max_gap):
    df = pd.DataFrame(columns=["TemporalPropertyID", "PAAWindowSize", "StdCoefficient", "MaxGap"])
    for ds in range(num_of_ds):
        new_row = pd.Series(data={"TemporalPropertyID": ds, "PAAWindowSize": paa_window_size, "StdCoefficient": std_coefficient,
                                  "MaxGap": max_gap})
        df = df.append(new_row, ignore_index=True)

        df.to_csv(path + '\\pp.csv', index=False)


# Create gkb file
def create_gkb_file(path, num_of_ds, number_of_bins):
    index = 0
    df = pd.DataFrame(columns=["StateID", "TemporalPropertyID", "Method", "BinID", "BinLow", "BinHigh", "BinLowScore"])

    bin_degree = 180 / number_of_bins

    for ds in range(num_of_ds):
        bin_low = -90

        for bin_id in range(number_of_bins):
            new_row = pd.Series(data={"StateID": index, "TemporalPropertyID": ds, "Method": "gradient",
                                      "BinID": bin_id, "BinLow": bin_low, "BinHigh": bin_low + bin_degree, "BinLowScore": ""})
            bin_low += bin_degree
            index += 1

            df = df.append(new_row, ignore_index=True)

    df.to_csv(path + '\\gkb.csv', index=False)

create_pp_file("C:\\Users\\Shaha\\Desktop\\TA\\TEST", 1, 1, -1, 1)
create_gkb_file("C:\\Users\\Shaha\\Desktop\\TA\\TEST", 1, 3)
create_ta_file("C:\\Users\\Shaha\\Desktop\\TA\\TEST", 1, "equal-frequency", 3)

