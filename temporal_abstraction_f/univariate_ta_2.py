import markdown
import re
import pandas as pd
import numpy as np

from utils_folder.utils import open_pickle
from utils_folder.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018


def new_uts_files(cur_root_dir):
    """
    :param datasets:
    :param cur_root_dir: the location in which all the databases are saved
    :return: the function create all the transformations
    """
    files_type = ["Train", "Test"]

    univariate_dict = open_pickle("univariate_dict")

    for index, dataset_name in enumerate(DATASET_NAMES_2018):
        root_dir_dataset = cur_root_dir + dataset_name

        transformation_dict = {"1": transformation_1, "2": transformation_2, "3": transformation_3}

        print("\t" + dataset_name + ":")

        for file_type in files_type:
            # Get from the read me file the number of rows, number of columns and number of
            classes = univariate_dict[(dataset_name, file_type.lower())]["classes"]
            number_of_rows = univariate_dict[(dataset_name, file_type.lower())]["rows"]
            number_of_columns = univariate_dict[(dataset_name, file_type.lower())]["columns"]

            # Run the three transformation on the Train and Test files
            for key in transformation_dict.keys():
                transformation_dict[key](root_dir_dataset, file_type, number_of_rows, number_of_columns, classes)
                print("\t\ttransformation_" + key + ", " + file_type.lower())

        print("")


def transformation_1(path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """

    # Create empty numpy array
    arr = np.zeros((number_of_rows, number_of_columns), int)

    for class_id in classes:
        # Read the hugobot output file for class_id
        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(int(class_id)) + ".txt"
        else:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                # Add the classifier column
                arr[int(entity_id), 0] = int(class_id)

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns
                    arr[int(entity_id), int(parse_data[0]): int(parse_data[1])] = parse_data[2]

    # Save the file
    pd.DataFrame(arr).to_csv(path + '//transformation2_type1_' + file_type + '.csv', index=False, header=None)


def transformation_2(path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "//output//" + file_type.lower() + "//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (entity, state), value: row in table
    for entity in range(number_of_rows):
        for state in states:
            rows_dict[(entity, state)] = index
            index += 1

    # Create empty numpy array
    arr = np.full((number_of_rows * number_of_states, number_of_columns), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(int(class_id)) + ".txt"
        else:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns

                    dict_value = rows_dict[(int(entity_id), int(parse_data[2]))]
                    arr[dict_value, int(parse_data[0]): int(parse_data[1])] = True

                    # Add the classifier column
                    arr[dict_value, 0] = int(class_id)

    df = pd.DataFrame(arr)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)

    # Save the file
    df.to_csv(path + '//transformation2_type2_' + file_type + '.csv', index=False, header=None)


def transformation_3(path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "//output//" + file_type.lower() + "//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (entity, state), value: row in table
    for entity in range(number_of_rows):
        for state in states:
            rows_dict[(entity, state, '+')] = index
            rows_dict[(entity, state, '-')] = index + 1
            index += 2

    # Create empty numpy array
    arr = np.full((number_of_rows * number_of_states * 2, number_of_columns), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        # TODO
        # ta_output = path + "\\output\\" + file_type.lower() + "\\KL-class-" + str(int(float(class_id))) + ".txt"

        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(int(class_id)) + ".txt"
        else:
            ta_output = path + "//output//" + file_type.lower() + "//KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns

                    dict_value_1 = rows_dict[(int(entity_id), int(parse_data[2]), '+')]
                    dict_value_2 = rows_dict[(int(entity_id), int(parse_data[2]), '-')]

                    arr[dict_value_1, int(parse_data[0])] = True
                    arr[dict_value_2, int(parse_data[0])] = True

                    # Add the classifier column
                    arr[dict_value_1, 0] = int(class_id)
                    arr[dict_value_2, 0] = int(class_id)

    df = pd.DataFrame(arr)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)

    # Save the file
    df.to_csv(path + '//transformation2_type3_' + file_type + '.csv', index=False, header=None)
