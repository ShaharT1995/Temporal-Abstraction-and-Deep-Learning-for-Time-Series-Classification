import pandas as pd
import numpy as np

from utils_folder.utils import open_pickle, create_directory


def new_mts_files(config, prop_path):
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

        path = prop_path + dataset_name + "/"
        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        transformation_dict = {"1": transformation_1, "2": transformation_2, "3": transformation_3}

        number_of_entities_train = mts_dict[dataset_name]["number_of_entities_train"]
        number_of_entities_test = mts_dict[dataset_name]["number_of_entities_test"]
        time_serious_length = mts_dict[dataset_name]["time_serious_length"]
        number_of_attributes = mts_dict[dataset_name]["number_of_attributes"]

        y = np.load(config.mts_path + dataset_name + '//y_train.npy')
        classes = np.unique(y)

        for file_type in files_type:
            # Run the three transformation on the Train and Test files
            if file_type == "train":
                transformation_dict[config.transformation_number](path, output_path, file_type,
                                                                  number_of_entities_train, time_serious_length,
                                                                  number_of_attributes, classes)
                print("\t\ttransformation_" + config.transformation_number + ", train")
            else:
                transformation_dict[config.transformation_number](path, output_path, file_type,
                                                                  number_of_entities_test, time_serious_length,
                                                                  number_of_attributes, classes)
                print("\t\ttransformation_" + config.transformation_number + ", test")
        print("")


def transformation_1(path, output_path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_entities: the number of entities in the database
    :param time_serious_length: the length of the time series
    :param number_of_attributes: the number of the time series in the database
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "train//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    min_property = int(min(states_df["TemporalPropertyID"]))

    # Create empty numpy array
    arr = np.zeros((number_of_entities, time_serious_length, number_of_attributes))

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + file_type + "//KL-class-" + str(float(class_id)) + ".txt"

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

                    # For the relevant entity, put the time series value in the relevant serious in the range of
                    # start_timestamp to the end_timestamp
                    arr[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, int(parse_data[3]) -
                                                                                        min_property] = parse_data[2]

    # Save the file
    np.save(output_path + 'type1_' + file_type + '.npy', arr)


def transformation_2(path, output_path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_entities: the number of entities in the database
    :param time_serious_length: the length of the time series
    :param number_of_attributes: not used in this function
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "train//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]

    # Create empty numpy array
    arr = np.full((number_of_entities, time_serious_length, number_of_states), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + file_type + "//KL-class-" + str(float(class_id)) + ".txt"

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

                    arr[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, int(parse_data[2]) - 1] = True

    # Save the file
    np.save(output_path + 'type2_' + file_type + '.npy', arr)


def transformation_3(path, output_path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    """
    :param path: the location of the hugobot output
    :param file_type: train/test
    :param number_of_entities: the number of entities in the database
    :param time_serious_length: the length of the time series
    :param number_of_attributes: not used in this function
    :param classes: the classes in the database
    :return: the function do the transformation and save the data after it
    """
    states_path = path + "train//states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (attribute, state), value: row in table
    for state in states:
        rows_dict[(state, '+')] = index
        rows_dict[(state, '-')] = index + 1
        index += 2

    # Create empty numpy array
    arr = np.full((number_of_entities, time_serious_length, number_of_states * 2), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + file_type + "//KL-class-" + str(float(class_id)) + ".txt"

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

                    # Get the row index of the (attribute, symbol)
                    dict_value_1 = rows_dict[(int(parse_data[2]), '+')]
                    dict_value_2 = rows_dict[(int(parse_data[2]), '-')]

                    arr[int(entity_id)][int(parse_data[0]) - 1][dict_value_1] = True
                    arr[int(entity_id)][int(parse_data[1]) - 2][dict_value_2] = True

    # Save the file
    np.save(output_path + 'type3_' + file_type + '.npy', arr)
