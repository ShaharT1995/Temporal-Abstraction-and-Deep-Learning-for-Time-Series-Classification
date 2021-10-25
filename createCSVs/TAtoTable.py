import markdown
import re
import pandas as pd
import numpy as np


def transformation_1(path, file_type):
    read_me_path = path + "\\README.md"

    # Read MD file
    read_me = markdown.markdown(open(read_me_path).read())

    # Get from the read me file the number of rows, number of columns and number of classes
    number_of_columns = int(re.search('Time series length: (.*)</p>', read_me).group(1)) + 1
    search_string = file_type + " size: "
    number_of_rows = int(re.search(search_string + '(.*)</p>', read_me).group(1))
    number_of_classes = int(re.search('Number of classses: (.*)</p>', read_me).group(1))

    # Create empty numpy array
    arr = np.zeros((number_of_rows, number_of_columns), int)

    for class_id in range(number_of_classes):
        # Read the hugobot output file for class_id
        ta_output = path + "\\output\\KL-class-" + str(float(class_id)) + ".txt"

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
    pd.DataFrame(arr).to_csv(path + '\\after_TA_' + file_type + '.csv', index=False, header=None)


def transformation_2(path, file_type):
    read_me_path = path + "\\README.md"
    states_path = path + "\\output\\states.csv"

    # Read MD file
    read_me = markdown.markdown(open(read_me_path).read())

    # Get from the read me file the number of rows, number of columns and number of classes
    number_of_columns = int(re.search('Time series length: (.*)</p>', read_me).group(1)) + 1
    search_string = file_type + " size: "
    number_of_rows = int(re.search(search_string + '(.*)</p>', read_me).group(1))
    number_of_classes = int(re.search('Number of classses: (.*)</p>', read_me).group(1))

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

    for class_id in range(number_of_classes):
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
                    # For each entity, put the state id in the range of start_time to end_time columns

                    dict_value = rows_dict[(int(entity_id), int(parse_data[2]))]
                    arr[dict_value, int(parse_data[0]): int(parse_data[1])] = True

                    # Add the classifier column
                    arr[dict_value, 0] = int(class_id)

    df = pd.DataFrame(arr)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)

    # Save the file
    df.to_csv(path + '\\after_boolean_TA_' + file_type + '.csv', index=False, header=None)


def transformation_3(path, file_type):
    read_me_path = path + "\\README.md"

    # Read MD file
    read_me = markdown.markdown(open(read_me_path).read())

    # Get from the read me file the number of rows, number of columns and number of classes
    number_of_columns = int(re.search('Time series length: (.*)</p>', read_me).group(1)) + 1
    search_string = file_type + " size: "
    number_of_rows = int(re.search(search_string + '(.*)</p>', read_me).group(1))
    number_of_classes = int(re.search('Number of classses: (.*)</p>', read_me).group(1))

    # Create empty df (one more to classifier and one to entity id)
    # df = pd.DataFrame(columns=[np.arange(number_of_columns), "EntityID", "Classifier"])
    df = pd.DataFrame(columns=np.arange(number_of_columns + 2), dtype=bool)

    for class_id in range(number_of_classes):
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

                    row_1 = pd.Series(data={number_of_columns: (entity_id, info), number_of_columns + 1: class_id,
                                            int(parse_data[0]): True})
                    row_2 = pd.Series(data={number_of_columns: (entity_id, info), number_of_columns + 1: class_id,
                                            int(parse_data[1]): True})

                    df = df.append(row_1, ignore_index=True)
                    df = df.append(row_2, ignore_index=True)
    # Save the file
    df.to_csv(path + '\\after_3_TA_' + file_type + '.csv', index=False, header=None)


transformation_3("C:\\Users\\Shaha\\Desktop\\UCRArchive_2018\\archives\\UCRArchive_2018\\Coffee", "Train")
