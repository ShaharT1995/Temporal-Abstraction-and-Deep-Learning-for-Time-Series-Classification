import numpy as np
import pandas as pd

NEXT_ATTRIBUTE_ID = 0


def input_to_csv(path, file_type):
    global NEXT_ATTRIBUTE_ID

    # Reading the npy files
    x = np.load(path + 'x_' + file_type + '.npy')
    y = np.load(path + 'y_' + file_type + '.npy')

    # Reshape the data frame
    m, n, r = x.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), x.reshape(m * n, -1)))

    # Array to DF
    out_df = pd.DataFrame(out_arr, columns=['EntityID'] + list(range(1, r + 1)))
    # Add the timestamp column to the df (the timestamp was the row index + 1)
    out_df['TimeStamp'] = np.arange(1, len(out_df) + 1)

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
    merged = merged[['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']]

    merged.to_csv(path + '2' + file_type + '.csv', index=False)

    # for entity_id in range(len(x)):
    #     tmp = x[entity_id]
    #     for attribute_id in range(len(tmp)):
    #         for timestamp_index in range(1, len(x[entity_id]) + 1):
    #             new_row = pd.Series(data={"EntityID": entity_id, "TimeStamp": timestamp_index, "TemporalPropertyID": NEXT_ATTRIBUTE_ID + attribute_id,
    #                                       "TemporalPropertyValue": tmp[attribute_id][timestamp_index]})
    #             df = df.append(new_row, ignore_index=True)
    #
    #             NEXT_ATTRIBUTE_ID += 1
    #
    # df_classifier = pd.DataFrame(y).reset_index().melt('index')
    # df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]
    #
    # # Adding the TemporalPropertyID with fix value
    # df_classifier["TemporalPropertyID"] = -1
    #
    # # Create classifier DF with numpy
    # merged = pd.concat([df, df_classifier])
    # merged = merged[['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']]
    #
    # merged.to_csv(path + '_TA' + file_type + '.csv', index=False)

def convert_all_df(cur_root_dir):
    MTS_DATASET_NAMES = ["ECG"]
    for dataset_name in MTS_DATASET_NAMES:
        root_dir_dataset = cur_root_dir + '/archives/mts_archive/' + dataset_name + '/'
        input_to_csv(root_dir_dataset, "train")
        input_to_csv(root_dir_dataset, "test")


convert_all_df("C:\\Users\\Shaha\\Desktop\\mtsdata")
