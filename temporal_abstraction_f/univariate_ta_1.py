import numpy as np
import pandas as pd

from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils_folder.utils import write_pickle

classes_dict = {}

def input_to_csv(path, file_type, property_id, dataset_name):
    global classes_dict

    # Reading the tsv file
    tsv_data = pd.read_csv(path + file_type + ".tsv", sep='\t', header=None)

    # Drop the classifier column
    df = tsv_data.drop([0], axis=1)

    # Create the new DF with numpy
    df = pd.DataFrame(df).reset_index().melt('index')
    df.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

    # Adding the TemporalPropertyID with fix value
    df["TemporalPropertyID"] = property_id

    # Drop all columns except classifier column
    classifier_data = tsv_data.values[:, 0]

    classes_dict[dataset_name] = np.unique(classifier_data)

    df_classifier = pd.DataFrame(classifier_data).reset_index().melt('index')
    df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

    # Adding the TemporalPropertyID with fix value
    df_classifier["TemporalPropertyID"] = -1

    # Create classifier DF with numpy
    merged = pd.concat([df, df_classifier])
    merged = merged[['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']]

    merged.to_csv(path + 'U-transformation1' + file_type + '.csv', index=False)


def convert_all_UTS(cur_root_dir, next_attribute):
    global classes_dict
    DATASET_NAMES_2018 = ["Coffee"]

    file_types = ["_TRAIN", "_TEST"]

    for dataset_name in DATASET_NAMES_2018:
        for file_type in file_types:
            root_dir_dataset = cur_root_dir + '/archives/UCRArchive_2018/' + dataset_name + '/' + dataset_name
            input_to_csv(root_dir_dataset, file_type, next_attribute, dataset_name)
        next_attribute += 1

    write_pickle("univariate_classes_dict", classes_dict)

    return next_attribute

#convert_all_df("C:\\Users\\Shaha\\Desktop\\UCRArchive_2018")
