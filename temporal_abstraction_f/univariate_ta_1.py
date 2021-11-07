import numpy as np
import pandas as pd

from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils_folder.utils import write_pickle


class UnivariateTA1:
    def __init__(self, cur_root_dir, next_attribute):
        """
        :param cur_root_dir: the location in which all the databases are saved
        :param next_attribute: the count of the time series
        """
        self.cur_root_dir = cur_root_dir
        self.next_attribute = next_attribute

        self.classes_dict = {}

    def input_to_csv(self, path, file_type, dataset_name):
        """
        :param path: the location of the original files
        :param file_type: the type of the file - train/test
        :param property_id: the index of time series
        :param dataset_name: the name of the dataset
        :return: the function create the csv file of the hugobot input format
        """
        # Reading the tsv file
        tsv_data = pd.read_csv(path + file_type + ".tsv", sep='\t', header=None)

        # Drop the classifier column
        df = tsv_data.drop([0], axis=1)

        # Create the new DF with numpy
        df = pd.DataFrame(df).reset_index().melt('index')
        df.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

        # Adding the TemporalPropertyID with fix value
        df["TemporalPropertyID"] = self.next_attribute

        # Drop all columns except classifier column
        classifier_data = tsv_data.values[:, 0]

        self.classes_dict[dataset_name] = np.unique(classifier_data)

        df_classifier = pd.DataFrame(classifier_data).reset_index().melt('index')
        df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

        # Adding the TemporalPropertyID with fix value
        df_classifier["TemporalPropertyID"] = -1

        # Create classifier DF with numpy
        merged = pd.concat([df, df_classifier])
        merged = merged[['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']]

        merged.to_csv(path + 'U-transformation1' + file_type + '.csv', index=False)

    def convert_all_UTS(self):
        """
        :param cur_root_dir: the location in which all the databases are saved
        :param next_attribute: the count of the time series
        :return: the function return the count of the time series
        """
        file_types = ["_TRAIN", "_TEST"]

        for dataset_name in DATASET_NAMES_2018:
            for file_type in file_types:
                root_dir_dataset = self.cur_root_dir + dataset_name + '/' + dataset_name
                self.input_to_csv(root_dir_dataset, file_type, dataset_name)
            self.next_attribute += 1

        write_pickle("univariate_classes_dict", self.classes_dict)

        return self.next_attribute
