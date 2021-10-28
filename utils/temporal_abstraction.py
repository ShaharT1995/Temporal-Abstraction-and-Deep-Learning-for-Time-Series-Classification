from temporal_abstraction_f.multivariate_ta_1 import convert_all_MTS
from temporal_abstraction_f.univariate_ta_1 import convert_all_UTS
from temporal_abstraction_f.set_parameters import create_three_files
from temporal_abstraction_f.multivariate_ta_2 import new_mts_files
from temporal_abstraction_f.univariate_ta_2 import new_uts_files


def run(path):
    # Make the first temporal abstraction -> original data sets to hugobot format
    convert_all_MTS(path + "mtsdata")
    convert_all_UTS(path + "UCRArchive_2018")

    # Make the 3 files - gkb.csv, ta.csv and ppa.csv
    create_three_files(path="C:\\Users\\Shaha\\Desktop\\TA\\TEST",
                       method="equal-frequency",
                       nb_bins=3,
                       paa_window_size=1,
                       std_coefficient=-1,
                       max_gap=-1)

    # Make the second temporal abstraction -> hugobot output files to original format
    new_mts_files(path + "mtsdata")
    new_uts_files(path + "UCRArchive_2018")


root_dir = "C:\\Users\\Shaha\\Desktop\\"
run(root_dir)


