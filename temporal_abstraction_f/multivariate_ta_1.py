import numpy as np
import pandas as pd

from utils_folder.constants import MTS_DATASET_NAMES


def input_to_csv(path, file_type, next_attribute):
    # Reading the npy files
    x = np.load(path + 'x_' + file_type + '.npy')
    y = np.load(path + 'y_' + file_type + '.npy')

    # Reshape the data frame
    m, n, r = x.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), x.reshape(m * n, -1)))

    timestamps = np.arange(1, n + 1)
    timestamps = np.tile(timestamps, (m, 1))
    timestamps = pd.DataFrame(timestamps.reshape(timestamps.shape[0] * timestamps.shape[1]))

    out_df = pd.DataFrame(out_arr, columns=['EntityID'] + list(range(next_attribute, r + next_attribute)))

    # Set the NEXT_ATTRIBUTE_ID variable
    next_attribute += r

    out_df["TimeStamp"] = timestamps

    # Set the DF to the input format
    df = out_df.melt(id_vars=['EntityID', 'TimeStamp'], ignore_index=False)
    df.columns = ["EntityID", "TimeStamp", "TemporalPropertyID", "TemporalPropertyValue"]
    # Change the column order
    df = df.reindex(columns=["EntityID", "TemporalPropertyID", "TimeStamp", "TemporalPropertyValue"])

    # Create classifier DF with numpy
    df_classifier = pd.DataFrame(y).reset_index().melt('index')
    df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

    # Adding the TemporalPropertyID with fix value
    df_classifier["TemporalPropertyID"] = -1

    # Merging between the two data frames
    merged = pd.concat([df, df_classifier])

    merged.to_csv(path + 'M-transformation1_' + file_type + '.csv', index=False)
    return next_attribute


def convert_all_MTS(cur_root_dir, next_attribute):
    MTS_DATASET_NAMES = ["ECG"]

    for dataset_name in MTS_DATASET_NAMES:
        root_dir_dataset = cur_root_dir + '/archives/mts_archive/' + dataset_name + '/'
        input_to_csv(root_dir_dataset, "train", next_attribute)
        next_attribute = input_to_csv(root_dir_dataset, "test", next_attribute)

    return next_attribute



#convert_all_MTS("C:\\Users\\Shaha\\Desktop\\mtsdata")
