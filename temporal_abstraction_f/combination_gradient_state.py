import pandas as pd
import numpy as np

from utils_folder.utils import open_pickle, create_directory


def combining_two_methods_ucr(config, prop_path, gradient_path):
    """
    :return: the function create all the transformations
    """
    files_type = ["Train", "Test"]

    univariate_dict = open_pickle("univariate_dict")

    for index, dataset_name in enumerate(config.UNIVARIATE_DATASET_NAMES_2018):
        path = prop_path + dataset_name + "//"
        gradient_dataset_path = gradient_path + dataset_name + "//"

        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        print("\t" + dataset_name + ":")

        states_path = path + "train//states.csv"
        states_df = pd.read_csv(states_path, header=0)

        states_path_gradient = gradient_dataset_path + "train//states.csv"
        states_df_gradient = pd.read_csv(states_path_gradient, header=0)

        for file_type in files_type:
            print("\t\t" + file_type)
            # Get from the read me file the number of rows, number of columns and number of
            classes = univariate_dict[(dataset_name, file_type.lower())]["classes"]
            number_of_entities = univariate_dict[(dataset_name, file_type.lower())]["rows"]
            time_serious_length = univariate_dict[(dataset_name, file_type.lower())]["columns"] - 1

            # Run the three transformation on the Train and Test files
            create_transformations(config, path, gradient_dataset_path, output_path, file_type, number_of_entities,
                                   time_serious_length, 1, classes, states_df,
                                   states_df_gradient)

        print("")


def combining_two_methods_mts(config, prop_path, gradient_path):
    """
    :return: the function create all the transformations
    """
    files_type = ["train", "test"]

    mts_dict = open_pickle("MTS_Dictionary")

    for index, dataset_name in enumerate(config.MTS_DATASET_NAMES):
        path = prop_path + dataset_name + "//"
        gradient_dataset_path = gradient_path + dataset_name + "//"

        output_path = config.path_transformation2 + dataset_name + "//"

        create_directory(output_path)

        print("\t" + dataset_name + ":")

        states_path = path + "train//states.csv"
        states_df = pd.read_csv(states_path, header=0)

        states_path_gradient = gradient_dataset_path + "train//states.csv"
        states_df_gradient = pd.read_csv(states_path_gradient, header=0)

        number_of_entities_train = mts_dict[dataset_name]["number_of_entities_train"]
        number_of_entities_test = mts_dict[dataset_name]["number_of_entities_test"]
        time_serious_length = mts_dict[dataset_name]["time_serious_length"]
        number_of_attributes = mts_dict[dataset_name]["number_of_attributes"]

        y = np.load(config.mts_path + dataset_name + '//y_train.npy')
        classes = np.unique(y)

        for file_type in files_type:
            print("\t\t" + file_type)
            # Run the three transformation on the Train and Test files
            if file_type == "train":
                create_transformations(config, path, gradient_dataset_path, output_path, file_type,
                                       number_of_entities_train, time_serious_length, number_of_attributes, classes,
                                       states_df, states_df_gradient)
            else:
                create_transformations(config, path, gradient_dataset_path, output_path, file_type,
                                       number_of_entities_test, time_serious_length, number_of_attributes, classes,
                                       states_df, states_df_gradient)

        print("")


def organize_df_per_entity(config, states_df, states_df_gradient, number_of_attributes, number_of_states):
    if config.archive == "UCR":
        states_df["TemporalPropertyID"] = 0
        states_df_gradient["TemporalPropertyID"] = 0
    else:
        states_df["TemporalPropertyID"] = states_df["TemporalPropertyID"] - (states_df["TemporalPropertyID"] // 10) * 10
        states_df_gradient["TemporalPropertyID"] = states_df_gradient["TemporalPropertyID"] - \
                                                   (states_df_gradient["TemporalPropertyID"] // 10) * 10

    states_df["StateID"] = states_df["StateID"] % int(number_of_states / number_of_attributes)
    states_df.loc[states_df["StateID"] == 0, "StateID"] = int(number_of_states / number_of_attributes)

    states_df_gradient["StateID"] = states_df_gradient["StateID"] % int(number_of_states / number_of_attributes)
    states_df_gradient.loc[states_df_gradient["StateID"] == 0, "StateID"] = int(number_of_states / number_of_attributes)

    states_df["StateID"] = states_df["StateID"] + states_df["TemporalPropertyID"] * (number_of_states /
                                                                                         number_of_attributes)
    states_df_gradient["StateID"] = states_df_gradient["StateID"] + states_df_gradient["TemporalPropertyID"] * \
                                    (number_of_states / number_of_attributes)

    return states_df, states_df_gradient


def create_transformations(config, path, gradient_path, output_path, file_type, number_of_entities, time_serious_length,
                           number_of_attributes, classes, states_df, states_df_gradient):
    number_of_states = states_df.shape[0]

    if config.perEntity:
        number_of_states = int(number_of_states / number_of_entities)
        states_df, states_df_gradient = organize_df_per_entity(config, states_df, states_df_gradient,
                                                               number_of_attributes, number_of_states)

    max_state = int(max(states_df["StateID"]))

    # Get the number of state from state.csv file- for gradient
    number_of_states = number_of_states * 2

    # For transformation 1
    min_property = int(min(states_df["TemporalPropertyID"]))

    # For transformation 3
    rows_dict = {}
    index = 0

    # Create key value for the output table -> key: (entity, state), value: row in table
    for state in range(1, number_of_states + 1):
        rows_dict[(state, '+')] = index
        rows_dict[(state, '-')] = index + 1
        index += 2

    # Create empty numpy array
    arr_1 = np.zeros((number_of_entities, time_serious_length, number_of_attributes * 2))
    arr_2 = np.full((number_of_entities, time_serious_length, number_of_states), False, dtype=bool)
    arr_3 = np.full((number_of_entities, time_serious_length, number_of_states * 2), False, dtype=bool)

    arr_1, arr_2, arr_3 = fill_transformations(config, arr_1, arr_2, arr_3, path, file_type, classes,
                                               number_of_attributes, 0, rows_dict, min_property, number_of_states, 0)

    arr_1, arr_2, arr_3 = fill_transformations(config, arr_1, arr_2, arr_3, gradient_path, file_type, classes,
                                               number_of_attributes, max_state, rows_dict, min_property,
                                               number_of_states, number_of_attributes)

    # Save the arrays
    np.save(output_path + 'type1_' + file_type.lower() + '_combination.npy', arr_1)
    np.save(output_path + 'type2_' + file_type.lower() + '_combination.npy', arr_2)
    np.save(output_path + 'type3_' + file_type.lower() + '_combination.npy', arr_3)


def fill_transformations(config, arr_1, arr_2, arr_3, path, file_type, classes, number_of_attributes, max_state,
                         rows_dict, min_property, number_of_states, start_from):
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

                for info in range(len(data) - 1):
                    # Extract the start time, end time and state id
                    parse_data = data[info].split(',')

                    if config.perEntity:
                        temporal_property_ID = 0 if config.archive == "UCR" else int(parse_data[3]) - \
                                                                                 (int(parse_data[3]) // 10) * 10

                        modulo = int(parse_data[2]) % int(number_of_states / (number_of_attributes * 2))
                        if modulo == 0:
                            modulo = int(number_of_states / (number_of_attributes * 2))

                        symbol = int(modulo + temporal_property_ID * (number_of_states / (number_of_attributes * 2)))

                    else:
                        temporal_property_ID = int(parse_data[3]) - min_property
                        symbol = int(parse_data[2])

                    # Transformation 1
                    arr_1[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, temporal_property_ID +
                                                                                          start_from] = symbol + \
                                                                                                        max_state

                    # Transformation 2
                    arr_2[int(entity_id)][int(parse_data[0]) - 1: int(parse_data[1]) - 1, symbol + max_state - 1] = True

                    # Transformation 3
                    dict_value_1 = rows_dict[(symbol + max_state, '+')]
                    dict_value_2 = rows_dict[(symbol + max_state, '-')]

                    arr_3[int(entity_id)][int(parse_data[0]) - 1][dict_value_1] = True
                    arr_3[int(entity_id)][int(parse_data[1]) - 2][dict_value_2] = True

    return arr_1, arr_2, arr_3

