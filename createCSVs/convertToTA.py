import pandas as pd
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018


def input_to_csv(path, file_type, property_id):
    input_data_frame = pd.DataFrame(columns=["EntityID", "TemporalPropertyID", "TimeStamp", "TemporalPropertyValue"])
    df = pd.read_csv(path + file_type + ".tsv", sep='\t', header=None)

    for entity_id, row in df.iterrows():
        for index_column in range(1, len(df.columns)):
            time_stamp_value = df[index_column].iloc[entity_id]
            new_row = pd.Series(data={"EntityID": entity_id, "TemporalPropertyID": property_id, "TimeStamp": index_column,
                                     "TemporalPropertyValue": time_stamp_value})
            input_data_frame = input_data_frame.append(new_row, ignore_index=True)

    input_data_frame.to_csv(path + '_TA' + file_type + '.csv', index=False)


def convert_all_df(cur_root_dir):
    for index, dataset_name in enumerate(DATASET_NAMES_2018):
        root_dir_dataset = cur_root_dir + '/archives/UCRArchive_2018/' + dataset_name + '/' + dataset_name
        input_to_csv(root_dir_dataset, "_TRAIN", index)
        input_to_csv(root_dir_dataset, "_TEST", index)

convert_all_df("C:\\Users\\Shaha\\Desktop\\UCRArchive_2018")

