import pandas as pd
import numpy as np

from utils_folder.utils import open_pickle, create_directory


def combining_two_methods(config, prop_path):
    """
    :return: the function create all the transformations
    """
    files_type = ["Train", "Test"]

    univariate_dict = open_pickle("univariate_dict")

    for index, dataset_name in enumerate(config.UNIVARIATE_DATASET_NAMES_2018):
        transformation_dict = {"1": transformation_1, "2": transformation_2, "3": transformation_3}

        path = prop_path + dataset_name + "//"
        gradient_path = config.path_files_for_TA + config.archive + "//" + config.classifier + "//gradient//" \
                        + dataset_name + "//"

        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        print("\t" + dataset_name + ":")

        for file_type in files_type:
            # Get from the read me file the number of rows, number of columns and number of
            classes = univariate_dict[(dataset_name, file_type.lower())]["classes"]
            number_of_rows = univariate_dict[(dataset_name, file_type.lower())]["rows"]
            number_of_columns = univariate_dict[(dataset_name, file_type.lower())]["columns"] -1

            # Run the three transformation on the Train and Test files
            for key in transformation_dict.keys():
                transformation_dict[key](path, gradient_path, output_path, file_type, number_of_rows, number_of_columns, classes)
                print("\t\ttransformation_" + key + ", " + file_type.lower())

        print("")


def fill_transformation1(arr, path, file_type, classes, time_series_number, max_state,output_path, arr_class=None):
    for class_id in classes:
        # For the first method
        ta_output = path + file_type.lower() + "//KL-class-"
        # Read the hugobot output file for class_id
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

                # TODO
                #Add the classifier column
                if arr_class is not None:
                    arr_class[int(entity_id)] = int(class_id)

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns
                    arr[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, time_series_number] = \
                        int(parse_data[2])+ max_state

    if arr_class is not None:
        np.save(output_path + 'type1_' + file_type.lower() + 'classes.npy', arr_class)

    return arr


def transformation_1(path, gradient_path, output_path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "//train//states.csv"
    states_df = pd.read_csv(states_path, header=0)
    states = states_df.iloc[:, 0]
    max_state= max(states)

    # Create empty numpy array
    arr = np.zeros((number_of_rows, number_of_columns, 2))
    arr_class = np.zeros(number_of_rows)

    arr = fill_transformation1(arr, path, file_type, classes, 0, 0,output_path, arr_class)
    arr = fill_transformation1(arr,  gradient_path, file_type, classes, 1, max_state, output_path)

    # Save the file
    np.save(output_path + 'type1_' + file_type.lower() + '.npy', arr)


def fill_transformation2(arr, path, file_type, classes, max_state, output_path, arr_class= None):
    for class_id in classes:
        # Read the hugobot output file for class_id
        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output = path + file_type.lower() + "//KL-class-" + str(int(class_id)) + ".txt"
        else:
            ta_output = path + file_type.lower() + "//KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")

                if arr_class is not None:
                    arr_class[int(entity_id)] = int(class_id)

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns

                    arr[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1,
                    int(parse_data[2])+ max_state - 1] = True

    if arr_class is not None:
        np.save(output_path + 'type2_' + file_type.lower() + '_classes.npy', arr_class)

    return arr



def transformation_2(path, gradient_path, output_path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "//train//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]
    max_state= max(states)

    states_path_gradient = gradient_path + "//train//states.csv"
    # Get the number of state from state.csv file- for gradient
    states_df_gradient = pd.read_csv(states_path_gradient, header=0)
    number_of_states_gradient = states_df_gradient.shape[0]
    states_gradient = states_df_gradient.iloc[:, 0]
    states_gradient = states_gradient + max_state

    number_of_states= number_of_states + number_of_states_gradient

    # Create empty numpy array
    arr = np.full((number_of_rows, number_of_columns, number_of_states), False, dtype=bool)
    arr_class = np.zeros(number_of_rows)

    arr = fill_transformation2(arr, path, file_type, classes, 0, output_path, arr_class)
    arr = fill_transformation2(arr, gradient_path, file_type, classes, max_state, output_path)

    # Save the file
    np.save(output_path + 'type2_' + file_type.lower() + '.npy', arr)


def fill_transformation3(arr, path, file_type, classes,rows_dict, max_state,output_path,  arr_class=None):
    for class_id in classes:
        # Read the hugobot output file for class_id
        if "Chinatown" in path or "HouseTwenty" in path:
            ta_output = path + file_type.lower() + "//KL-class-" + str(int(class_id)) + ".txt"
        else:
            ta_output = path + file_type.lower() + "//KL-class-" + str(float(class_id)) + ".txt"

        with open(ta_output) as file:
            lines = file.readlines()
            for index in range(2, len(lines), 2):
                # Extract the entity id
                entity_id = lines[index][: len(lines[index]) - 2]
                # Extract the line of data
                data = lines[index + 1].split(";")
                if arr_class is not None:
                    arr_class[int(entity_id)] = int(class_id)

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')
                    # For each entity, put the state id in the range of start_time to end_time columns

                    dict_value_1 = rows_dict[(int(parse_data[2])+max_state, '+')]
                    dict_value_2 = rows_dict[(int(parse_data[2])+max_state, '-')]

                    arr[int(entity_id)][int(parse_data[0]) - 1][dict_value_1] = True
                    arr[int(entity_id)][int(parse_data[1]) - 2][dict_value_2] = True

    if arr_class is not None:
        np.save(output_path + 'type3_' + file_type.lower() + '_classes.npy', arr_class)

    return arr


def transformation_3(path, gradient_path, output_path, file_type, number_of_rows, number_of_columns, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_rows: the number of entities
    :param number_of_columns: the length of time series
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "train//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]
    max_state= max(states)

    states_path_gradient = gradient_path + "train//states.csv"

    # Get the number of state from state.csv file
    states_df_gradient = pd.read_csv(states_path_gradient, header=0)
    number_of_states_gradient = states_df_gradient.shape[0]
    states_gradient = states_df_gradient.iloc[:, 0]
    states_gradient = states_gradient + max_state

    states= pd.concat([states, states_gradient])
    number_of_states = number_of_states + number_of_states_gradient

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (entity, state), value: row in table
    for state in states:
        rows_dict[(state, '+')] = index
        rows_dict[(state, '-')] = index + 1
        index += 2

    # Create empty numpy array
    arr = np.full((number_of_rows, number_of_columns, number_of_states * 2), False, dtype=bool)
    arr_class = np.zeros(number_of_rows)
    arr = fill_transformation3(arr, path, file_type, classes, rows_dict, 0,output_path, arr_class)
    arr = fill_transformation3(arr, gradient_path, file_type, classes, rows_dict, max_state, output_path, arr_class)

    np.save(output_path + 'type3_' + file_type.lower() + '.npy', arr)