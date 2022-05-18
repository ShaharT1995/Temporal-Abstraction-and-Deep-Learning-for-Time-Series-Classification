import numpy as np
import pandas as pd

from utils_folder.utils import write_pickle, create_directory


class MultivariateTA1:
    def __init__(self, config, next_attribute):
        """
        :param cur_root_dir: the location in which all the databases are saved
        :param next_attribute: the count of the time series
        """
        self.config = config
        self.cur_root_dir = self.config.mts_path
        self.attributes_dict = {}

    def input_to_csv(self, path, dataset_name, file_type, output_path):
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

        out_df = pd.DataFrame(out_arr, columns=['EntityID'] + list(range(0, r)))

        out_df["TimeStamp"] = timestamps

        # Set the DF to the input format
        df = out_df.melt(id_vars=['EntityID', 'TimeStamp'], ignore_index=False)
        df.columns = ["EntityID", "TimeStamp", "TemporalPropertyID", "TemporalPropertyValue"]
        # Change the column order
        df = df.reindex(columns=["EntityID", "TemporalPropertyID", "TimeStamp", "TemporalPropertyValue"])

        df["TemporalPropertyID"] = (((df["EntityID"] + 1).astype(int)).astype(str) + "00" +
                                    (df["TemporalPropertyID"] + 1).astype(str)).astype(int)

        # Create classifier DF with numpy
        df_classifier = pd.DataFrame(y).reset_index().melt('index')
        df_classifier.columns = ["EntityID", "TimeStamp", "TemporalPropertyValue"]

        # Adding the TemporalPropertyID with fix value
        df_classifier["TemporalPropertyID"] = -1

        # Merging between the two data frames
        merged = pd.concat([df, df_classifier])

        create_directory(output_path)

        merged.to_csv(output_path + '/transformation1_' + file_type + '.csv', index=False)

        return pd.unique(df["TemporalPropertyID"])

    def convert_all_MTS(self):
        """
        :return: the function return the count of the time series
        """
        for dataset_name in self.config.MTS_DATASET_NAMES:
            print("\t" + dataset_name + ":")
            output_path = self.config.path_transformation1 + dataset_name + "//"
            root_dir_dataset = self.cur_root_dir + dataset_name + '/'

            print("\t\tTrain")
            self.attributes_dict[(dataset_name, "train")] = self.input_to_csv(root_dir_dataset, dataset_name, "train",
                                                                              output_path)
            print("\t\tTest")
            self.attributes_dict[(dataset_name, "test")] = self.input_to_csv(root_dir_dataset, dataset_name, "test",
                                                                              output_path)

        write_pickle("attributes_dict_mts_per_entity", self.attributes_dict)
