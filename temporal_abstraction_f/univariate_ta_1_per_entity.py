import os

import numpy as np
import pandas as pd

from utils_folder.utils import write_pickle, create_directory


class UnivariateTA1:
    def __init__(self, config, next_attribute):
        """
        :param cur_root_dir: the location in which all the databases are saved
        :param next_attribute: the count of the time series
        """
        self.config = config
        self.cur_root_dir = config.ucr_path
        self.next_attribute = next_attribute

        self.univariate_dict = {}

    def input_to_csv(self, path, file_type, dataset_name, output_path):
        """
        :param path: the location of the original files
        :param file_type: the type of the file - train/test
        :param property_id: the index of time series
        :param dataset_name: the name of the dataset
        :return: the function create the csv file of the hugobot input format
        """
        normalization = self.config.normalization

        # Reading the tsv file
        tsv_data = pd.read_csv(path + "_" + file_type + ".tsv", sep='\t', header=None)

        # Drop the classifier column
        df = tsv_data.drop([0], axis=1)

        # Z-Normalization
        if normalization:
            df.columns = range(df.shape[1])
            df = df.values

            std_ = df.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            df = (df - df.mean(axis=1, keepdims=True)) / std_

        # Create the new DF with numpy
        df = pd.DataFrame(df).reset_index().melt('index')
        df.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

        # Adding the TemporalPropertyID with fix value
        df["TemporalPropertyID"] = self.next_attribute

        # For the per entity
        df["TemporalPropertyID"] = df["TemporalPropertyID"] + df["EntityID"]

        # Drop all columns except classifier column
        classifier_data = tsv_data.values[:, 0]

        self.univariate_dict[(dataset_name, file_type.lower())] = {"classes": np.unique(classifier_data),
                                                                   "rows": tsv_data.shape[0],
                                                                   "columns": tsv_data.shape[1]}

        df_classifier = pd.DataFrame(classifier_data).reset_index().melt('index')
        df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

        # Adding the TemporalPropertyID with fix value
        df_classifier["TemporalPropertyID"] = -1

        # Create classifier DF with numpy
        merged = pd.concat([df, df_classifier])
        merged = merged[['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']]

        create_directory(output_path)

        merged.to_csv(output_path + "/transformation1_" + file_type.lower() + '.csv', index=False)

        return max(df["EntityID"]) + 1

    def convert_all_UCR(self):
        """
        :return: the function return the count of the time series
        """
        file_types = ["TRAIN", "TEST"]

        attributes_dict = {}

        for dataset_name in self.config.UNIVARIATE_DATASET_NAMES_2018:
            print("\t" + dataset_name + ":")
            for file_type in file_types:
                output_path = self.config.path_transformation1 + dataset_name + "//"
                root_dir_dataset = self.cur_root_dir + dataset_name + '/' + dataset_name

                r = self.input_to_csv(root_dir_dataset, file_type, dataset_name, output_path)

                attributes_dict[(dataset_name, file_type.lower())] = [i for i in range(self.next_attribute,
                                                                                       self.next_attribute + r)]

                print("\t\t" + file_type.lower())

            self.next_attribute += r

        write_pickle("univariate_dict", self.univariate_dict)
        write_pickle("attributes_dict_ucr_per_entity", attributes_dict)
