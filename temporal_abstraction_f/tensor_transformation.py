import os

import pandas as pd
import numpy as np

from utils_folder.utils import open_pickle, create_directory


def new_ucr_files(config, prop_path):
    """
    :return: the function create all the transformations
    """
    files_type = ["Train", "Test"]

    univariate_dict = open_pickle("univariate_dict")

    for index, dataset_name in enumerate(config.UNIVARIATE_DATASET_NAMES_2018):
        path = prop_path + dataset_name + "//"
        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        print("\t" + dataset_name)

        for file_type in files_type:
            states_path = path + file_type.lower() + "//states.csv"
            states_df = pd.read_csv(states_path, header=0)

            # Get from the read me file the number of rows, number of columns and number of
            classes = univariate_dict[(dataset_name, file_type.lower())]["classes"]
            number_of_rows = univariate_dict[(dataset_name, file_type.lower())]["rows"]
            number_of_columns = univariate_dict[(dataset_name, file_type.lower())]["columns"]

            classification_path = config.ucr_path + '/' + dataset_name + '/y_' + file_type.lower() + '.npy'

            # Run the three transformation on the Train and Test files
            create_transformations(config, path, output_path, file_type, number_of_rows, number_of_columns, 1, classes,
                                   states_df, univariate=True, classification_path=classification_path)
    print("")


def new_mts_files(config, prop_path, nb_bins):
    """
    :param cur_root_dir: the location in which all the databases are saved
    :return: the function create all the transformations
    """
    files_type = ["train", "test"]

    # Dictionary that contains the following data for each database: number_of_entities_train, number_of_entities_test
    # time_serious_length and number_of_attributes
    mts_dict = open_pickle("MTS_Dictionary")

    for index, dataset_name in enumerate(config.MTS_DATASET_NAMES):
        print("\t" + dataset_name)

        dataset_with_empty_columns = False

        path = prop_path + dataset_name + "/"
        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        number_of_entities_train = mts_dict[dataset_name]["number_of_entities_train"]
        number_of_entities_test = mts_dict[dataset_name]["number_of_entities_test"]
        time_serious_length = mts_dict[dataset_name]["time_serious_length"]
        number_of_attributes = mts_dict[dataset_name]["number_of_attributes"]

        y = np.load(config.mts_path + dataset_name + '//y_train.npy')
        classes = np.unique(y)

        for file_type in files_type:
            states_path = path + file_type + "//states.csv"
            states_df = pd.read_csv(states_path, header=0)
            # ------------------------------------------------------------
            # TODO - Alise (For our problem)
            if config.perEntity:
                count_df = pd.DataFrame(states_df['TemporalPropertyID'].value_counts())
                count_df = count_df.reset_index(level=0)
                count_df = count_df.loc[count_df['TemporalPropertyID'] < nb_bins]

                states_df['StateID'] = states_df['StateID'].astype(dtype='float')

                if count_df.shape[0] > 0:
                    dataset_with_empty_columns = True

                    for index, row in count_df.iterrows():
                        rows_to_add = nb_bins - row["TemporalPropertyID"]

                        next_state = states_df.loc[states_df["TemporalPropertyID"] == row["index"]]["StateID"].iloc[0] +\
                                     0.001

                        for i in range(rows_to_add):
                            states_df = states_df.append({"StateID": next_state,
                                                          "TemporalPropertyID": row["index"],
                                                          "BinID": i + 1,
                                                          "BinLow": float('-inf'),
                                                          "BinHigh": float('inf'),
                                                          "Method": config.method.upper()}, ignore_index=True)
                            next_state += 0.001

                    states_df = states_df.sort_values(by=['StateID', 'BinID'])
                    states_df.reset_index(drop=True, inplace=True)
                    states_df["index"] = states_df.index + 1
            # ------------------------------------------------------------

            # Run the three transformation on the Train and Test files
            if file_type == "train":
                create_transformations(config, path, output_path, file_type, number_of_entities_train,
                                       time_serious_length, number_of_attributes, classes, states_df,
                                       dataset_with_empty_columns=dataset_with_empty_columns)
            else:
                create_transformations(config, path, output_path, file_type, number_of_entities_test,
                                       time_serious_length, number_of_attributes, classes, states_df,
                                       dataset_with_empty_columns=dataset_with_empty_columns)
        print("")


def create_transformations(config, path, output_path, file_type, number_of_entities, time_serious_length,
                           number_of_attributes, classes, states_df, univariate=False, classification_path="",
                           dataset_with_empty_columns=False):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_entities: the number of entities in the database
    :param time_serious_length: the length of the time series
    :param number_of_attributes: not used in this function
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    # Get the number of state from state.csv file
    number_of_states = states_df.shape[0]

    if config.perEntity:
        number_of_states = int(number_of_states / number_of_entities)

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (attribute, state), value: row in table
    for state in range(1, number_of_states + 1):
        rows_dict[(state, '+')] = index
        rows_dict[(state, '-')] = index + 1
        index += 2

    # For transformation 1
    min_property = int(min(states_df["TemporalPropertyID"]))

    # Create empty numpy array
    arr_1 = np.zeros((number_of_entities, time_serious_length, number_of_attributes))
    arr_2 = np.full((number_of_entities, time_serious_length, number_of_states), False, dtype=bool)
    arr_3 = np.full((number_of_entities, time_serious_length, number_of_states * 2), False, dtype=bool)

    save_classification = False
    if univariate and not os.path.exists(classification_path):
        # Create the classification array
        arr_class = np.zeros(number_of_entities)
        save_classification = True

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + file_type.lower() + "//KL-class-"
        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output += str(int(class_id)) + ".txt"
        else:
            ta_output += str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                if save_classification:
                    arr_class[int(entity_id)] = int(class_id)

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')

                    # ------------------------------------------------------------
                    # TODO - Alisa
                    if config.perEntity:
                        if univariate:
                            temporal_property_ID = 0
                        else:
                            temporal_property_ID = int(parse_data[3][-1]) - 1 if parse_data[3][-2] == "0" else \
                                int(parse_data[3][-2:]) - 1

                        state_id = int(parse_data[2])
                        if dataset_with_empty_columns:
                            state_id = states_df["index"].loc[states_df["StateID"] == state_id].iloc[0]

                        modulo = state_id % int(number_of_states / number_of_attributes)
                        if modulo == 0:
                            modulo = int(number_of_states / number_of_attributes)

                        symbol = int(modulo + temporal_property_ID * (number_of_states / number_of_attributes))
                    # ------------------------------------------------------------

                    else:
                        temporal_property_ID = int(parse_data[3]) - min_property
                        symbol = int(parse_data[2])

                    # Get the row index of the (attribute, symbol)
                    dict_value_1 = rows_dict[(symbol, '+')]
                    dict_value_2 = rows_dict[(symbol, '-')]

                    arr_1[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, temporal_property_ID] = \
                        symbol

                    arr_2[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, symbol - 1] = True

                    arr_3[int(entity_id)][int(parse_data[0]) - 1][dict_value_1] = True
                    arr_3[int(entity_id)][int(parse_data[1]) - 2][dict_value_2] = True

    # Save the file
    np.save(output_path + 'type1_' + file_type.lower() + '.npy', arr_1)
    np.save(output_path + 'type2_' + file_type.lower() + '.npy', arr_2)
    np.save(output_path + 'type3_' + file_type.lower() + '.npy', arr_3)

    if save_classification:
        np.save(classification_path, arr_class)

