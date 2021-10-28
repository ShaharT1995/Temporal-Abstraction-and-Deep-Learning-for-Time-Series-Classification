import pandas as pd
import numpy as np

from utils.constants import MTS_DICT
from utils.constants import MTS_DATASET_NAMES


def new_mts_files(cur_root_dir):
    MTS_DATASET_NAMES = ["ECG"]

    files_type = ["train", "test"]

    for index, dataset_name in enumerate(MTS_DATASET_NAMES):
        root_dir_dataset = cur_root_dir + '/archives/mts_archive/' + dataset_name + "/"

        transformation_dict = {"1": transformation_1, "2": transformation_2, "3": transformation_3}

        number_of_entities_train = MTS_DICT[dataset_name]["number_of_entities_train"]
        number_of_entities_test = MTS_DICT[dataset_name]["number_of_entities_test"]
        time_serious_length = MTS_DICT[dataset_name]["time_serious_length"]
        number_of_attributes = MTS_DICT[dataset_name]["number_of_attributes"]

        y = np.load(root_dir_dataset + 'y_' + file_type + '.npy')
        classes = np.unique(y)

        for file_type in files_type:
            # Run the three transformation on the Train and Test files
            for key in transformation_dict.keys():
                if file_type == "train":
                    transformation_dict[key](root_dir_dataset, file_type, number_of_entities_train, time_serious_length,
                                             number_of_attributes, classes)
                else:
                    transformation_dict[key](root_dir_dataset, file_type, number_of_entities_test, time_serious_length,
                                             number_of_attributes, classes)


def transformation_1(path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    # Create empty numpy array
    arr = np.zeros((number_of_entities, number_of_attributes, time_serious_length))

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + "\\output\\KL-class-" + str(float(class_id)) + ".txt"

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
                    arr[int(entity_id)][int(parse_data[3])][int(parse_data[0]) - 1: int(parse_data[1]) - 1] = parse_data[2]

    # Save the file
    np.save(path + '\\after_TA-1_' + file_type + '.npy', arr)


def transformation_2(path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    states_path = path + "\\output\\states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]

    rows_dict = {}
    index = 0
    # Create key value for the output table -> key: (attribute, state), value: row in table
    for attribute in range(number_of_attributes):
        for state in states:
            rows_dict[(attribute, state)] = index
            index += 1

    # Create empty numpy array
    arr = np.full((number_of_entities, number_of_attributes * number_of_states, time_serious_length), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + "\\output\\KL-class-" + str(float(class_id)) + ".txt"

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
                    dict_value = rows_dict[(int(parse_data[3]), int(parse_data[2]))]

                    arr[int(entity_id)][dict_value][int(parse_data[0]) - 1: int(parse_data[1]) - 1] = True

    # Save the file
    np.save(path + '\\after_TA-2_' + file_type + '.npy', arr)


def transformation_3(path, file_type, number_of_entities, time_serious_length, number_of_attributes, classes):
    states_path = path + "\\output\\states.csv"

    # Get the number of state from state.csv file
    states_df = pd.read_csv(states_path, header=0)
    number_of_states = states_df.shape[0]
    states = states_df.iloc[:, 0]

    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (attribute, state), value: row in table
    for attribute in range(number_of_attributes):
        for state in states:
            rows_dict[(attribute, state, '+')] = index
            rows_dict[(attribute, state, '-')] = index + 1
            index += 2

    # Create empty numpy array
    arr = np.full((number_of_entities, number_of_attributes * number_of_states * 2, time_serious_length), False, dtype=bool)

    for class_id in classes:
        # Read the hugobot output file for class_id
        ta_output = path + "\\output\\KL-class-" + str(float(class_id)) + ".txt"

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
                    dict_value_1 = rows_dict[(int(parse_data[3]), int(parse_data[2]), '+')]
                    dict_value_2 = rows_dict[(int(parse_data[3]), int(parse_data[2]), '-')]

                    arr[int(entity_id)][dict_value_1][int(parse_data[0]) - 1] = True
                    arr[int(entity_id)][dict_value_2][int(parse_data[1]) - 2] = True

    # Save the file
    np.save(path + '\\after_TA-4_' + file_type + '.npy', arr)


# new_mts_files("C:\\Users\\Shaha\\Desktop\\mtsdata")