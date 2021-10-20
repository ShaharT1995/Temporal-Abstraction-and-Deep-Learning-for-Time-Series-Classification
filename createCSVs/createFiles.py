import pandas as pd


# Create TA file
def create_ta_file(path, num_of_ds, method, nb_bins, gradient_window_size =""):
    df = pd.DataFrame(columns=["Method", "NbBins", "GradientWindowSize"])
    for ds in range(num_of_ds):
        new_row = pd.Series(data={"TemporalPropertyID": ds, "Method": method, "NbBins": nb_bins,
                                  "GradientWindowSize": gradient_window_size})
        df = df.append(new_row, ignore_index=True)

    df.to_csv(path + '\\ta.csv', index=False)


# Create PP file
def create_pp_file(path, num_of_ds, paa_window_size, std_coefficient, max_gap):
    df = pd.DataFrame(columns=["PAAWindowSize", "StdCoefficient", "MaxGap"])
    for ds in range(num_of_ds):
        new_row = pd.Series(data={"PAAWindowSize": paa_window_size, "StdCoefficient": std_coefficient,
                                  "MaxGap": max_gap})
        df = df.append(new_row, ignore_index=True)

        df.to_csv(path + '\\pp.csv', index=False)


# def create_gkb_file(path, num_of_ds, number_of_bins, method, std_coefficient, max_gap):
#     df = pd.DataFrame(columns=["PAAWindowSize", "StdCoefficient", "MaxGap"])
#     for ds in range(num_of_ds):
#         new_row = pd.Series(data={"PAAWindowSize": paa_window_size, "StdCoefficient": std_coefficient,
#                                   "MaxGap": max_gap})
#         df = df.append(new_row, ignore_index=True)
#
#         df.to_csv(path + '\\pp.csv', index=False)


#create_ta_file ("C:\\Users\Shaha\Desktop\TA\TEST", 1, "method", 2)

