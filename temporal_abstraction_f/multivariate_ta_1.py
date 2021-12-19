import numpy as np
import pandas as pd

from utils_folder.constants import MTS_DATASET_NAMES
from utils_folder.utils import write_pickle


class MultivariateTA1:
    def __init__(self, cur_root_dir, next_attribute, attributes_dict):
        """
        :param cur_root_dir: the location in which all the databases are saved
        :param next_attribute: the count of the time series
        """
        self.cur_root_dir = cur_root_dir
        self.next_attribute = next_attribute
        self.attributes_dict = attributes_dict

    def input_to_csv(self, path, dataset_name, file_type):
        """
        :param path: the location of the original files
        :param file_type: the type of the file - train/test
        :return: the function create the csv file of the hugobot input format
        """
        # Reading the npy files
        x = np.load(path + 'x_' + file_type + '.npy')
        y = np.load(path + 'y_' + file_type + '.npy')

        # Reshape the data frame
        m, n, r = x.shape
        out_arr = np.column_stack((np.repeat(np.arange(m), n), x.reshape(m * n, -1)))

        timestamps = np.arange(1, n + 1)
        timestamps = np.tile(timestamps, (m, 1))
        timestamps = pd.DataFrame(timestamps.reshape(timestamps.shape[0] * timestamps.shape[1]))

        out_df = pd.DataFrame(out_arr, columns=['EntityID'] + list(range(self.next_attribute, r + self.next_attribute)))

        # Set the NEXT_ATTRIBUTE_ID variable
        if dataset_name not in self.attributes_dict.keys():
            self.attributes_dict[dataset_name] = [i for i in range(self.next_attribute, self.next_attribute + r)]
            self.next_attribute += r

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

    def convert_all_MTS(self):
        """
        :return: the function return the count of the time series
        """
        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = self.cur_root_dir + dataset_name + '/'
            self.input_to_csv(root_dir_dataset, dataset_name, "train")
            self.input_to_csv(root_dir_dataset, dataset_name, "test")

        write_pickle("attributes_dict", self.attributes_dict)

        return self.next_attribute
